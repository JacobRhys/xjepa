"""GPU-resident corpus storage and bucketed, token-budget batching.

The whole corpus (uint8 tokens, int32 offsets, fp16 [N, 128] target bank) is
under 3 GB and is moved to the device **once** at startup.  After
:meth:`GpuCorpus.load` returns there are no host->device transfers whatsoever:
batching is pure index arithmetic executed with torch ops on device.

There is deliberately no ``torch.utils.data.DataLoader``, no worker process, no
``pin_memory``, no prefetch stream and no collate function anywhere in this
module.  See ``docs/CONTRACTS.md``.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import torch

__all__ = ["Batch", "GpuCorpus", "BucketBatcher", "DEFAULT_BUCKETS", "TOKEN_BUDGET"]

# Length buckets, per contract.
DEFAULT_BUCKETS: tuple[int, ...] = (128, 256, 384, 512)
# Target tokens per optimiser step (B * L held ~constant across buckets).
TOKEN_BUDGET: int = 65_536

# ESM-2 alphabet constants (vocab = 33).
PAD_ID: int = 1
MASK_ID: int = 32


@dataclass
class Batch:
    """Fixed-shape batch, entirely on device.

    Attributes:
        tokens: int64 ``[B, L]`` model input (already corrupted if a masker ran).
        targets: float16 ``[B, L, target_dim]`` structure-embedding targets,
            zeroed at padding positions.
        pad_mask: bool ``[B, L]``, ``True`` marks a real residue.
        mask_sel: bool ``[B, L]``, ``True`` marks a masked position.
        labels: int64 ``[B, L]``, original token id where masked, ``-100``
            elsewhere.
        bucket: the bucket length ``L`` this batch was drawn from.
    """

    tokens: torch.Tensor
    targets: torch.Tensor
    pad_mask: torch.Tensor
    mask_sel: torch.Tensor
    labels: torch.Tensor
    bucket: int

    @property
    def batch_size(self) -> int:
        """Number of sequences in the batch (static per bucket)."""
        return int(self.tokens.shape[0])

    def to(self, device: str | torch.device) -> "Batch":
        """Return a copy of this batch on ``device`` (debug/eval use only)."""
        return Batch(
            tokens=self.tokens.to(device),
            targets=self.targets.to(device),
            pad_mask=self.pad_mask.to(device),
            mask_sel=self.mask_sel.to(device),
            labels=self.labels.to(device),
            bucket=self.bucket,
        )


def _copy_to_device(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    chunk_rows: int = 1 << 20,
) -> torch.Tensor:
    """Copy a (possibly memory-mapped) numpy array to device in row chunks.

    Chunking keeps peak host RAM at ``chunk_rows`` rows rather than the whole
    array, so a memory-mapped 2.6 GB target bank never needs to be materialised
    in RAM in one piece.

    Args:
        array: source array, typically ``np.load(..., mmap_mode="r")``.
        device: destination device.
        dtype: destination torch dtype.
        chunk_rows: number of leading-dimension rows per transfer.

    Returns:
        A device tensor with ``array``'s shape and ``dtype``.
    """
    shape = tuple(int(s) for s in array.shape)
    out = torch.empty(shape, dtype=dtype, device=device)
    n = shape[0] if shape else 0
    if n == 0:
        return out
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = np.array(array[start:stop], copy=True)
        out[start:stop].copy_(torch.from_numpy(block).to(dtype))
    return out


class GpuCorpus:
    """Whole corpus resident on device.  Nothing is transferred per step.

    Attributes:
        tokens: uint8 ``[total_residues]`` flat concatenation of all sequences.
        offsets: int32 ``[n_seqs + 1]`` start index of each sequence, with the
            total residue count as the final element.
        targets: float16 ``[total_residues, target_dim]`` PCA-reduced
            ESM-IF1 per-residue embeddings, row-aligned with ``tokens``.
        lengths: int32 ``[n_seqs]`` sequence lengths (``diff(offsets)``).
    """

    tokens: torch.Tensor
    offsets: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor

    def __init__(
        self,
        tokens: torch.Tensor,
        offsets: torch.Tensor,
        targets: torch.Tensor,
        meta: Optional[dict] = None,
    ) -> None:
        """Wrap already-on-device tensors.  Prefer :meth:`load`.

        Args:
            tokens: uint8 ``[total_residues]`` device tensor.
            offsets: int32 ``[n_seqs + 1]`` device tensor.
            targets: float16 ``[total_residues, target_dim]`` device tensor.
            meta: optional free-form metadata recorded alongside the cache.

        Raises:
            ValueError: if the shapes or dtypes are inconsistent.
        """
        if tokens.dim() != 1:
            raise ValueError(f"tokens must be 1-D, got shape {tuple(tokens.shape)}")
        if targets.dim() != 2:
            raise ValueError(f"targets must be 2-D, got shape {tuple(targets.shape)}")
        if offsets.dim() != 1 or offsets.numel() < 2:
            raise ValueError("offsets must be 1-D with at least 2 elements")
        if tokens.shape[0] != targets.shape[0]:
            raise ValueError(
                f"tokens ({tokens.shape[0]}) and targets ({targets.shape[0]}) "
                "must have the same number of residues"
            )
        if tokens.device != offsets.device or tokens.device != targets.device:
            raise ValueError("all corpus tensors must live on the same device")

        self.tokens = tokens
        self.offsets = offsets.to(torch.int32)
        self.targets = targets
        self.lengths = (self.offsets[1:] - self.offsets[:-1]).to(torch.int32)
        self.meta: dict = dict(meta or {})
        self._validate()

    def _validate(self) -> None:
        """Check offsets are monotone, in range, and cover the token array."""
        if bool((self.lengths < 0).any()):
            raise ValueError("offsets must be non-decreasing")
        if int(self.offsets[0]) != 0:
            raise ValueError("offsets[0] must be 0")
        if int(self.offsets[-1]) != int(self.tokens.shape[0]):
            raise ValueError(
                f"offsets[-1] ({int(self.offsets[-1])}) must equal total residues "
                f"({int(self.tokens.shape[0])})"
            )

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str, device: str, target_dim: int) -> "GpuCorpus":
        """Load a cache from disk and move it to ``device`` once.

        Two on-disk layouts are supported:

        * a directory holding ``tokens.npy`` (uint8), ``offsets.npy`` (int32),
          ``targets.npy`` (float16 ``[N, target_dim]``) and an optional
          ``meta.json``.  The arrays are memory-mapped and copied to device in
          chunks, so host RAM never holds the whole bank.
        * a ``.safetensors`` file with keys ``tokens``, ``offsets``,
          ``targets``.

        Args:
            path: directory or ``.safetensors`` file.
            device: torch device string, e.g. ``"cuda"`` or ``"cpu"``.
            target_dim: expected target width (128 in this project).

        Returns:
            A :class:`GpuCorpus` whose tensors are all resident on ``device``.

        Raises:
            FileNotFoundError: if the cache is missing.
            ValueError: if ``targets`` does not have width ``target_dim``.
        """
        dev = torch.device(device)
        if path.endswith(".safetensors"):
            tokens, offsets, targets, meta = cls._read_safetensors(path, dev)
        else:
            tokens, offsets, targets, meta = cls._read_npy_dir(path, dev)

        if int(targets.shape[1]) != int(target_dim):
            raise ValueError(
                f"target bank width {int(targets.shape[1])} != expected target_dim "
                f"{target_dim}"
            )
        return cls(tokens=tokens, offsets=offsets, targets=targets, meta=meta)

    @staticmethod
    def _read_npy_dir(
        path: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Memory-map a ``.npy`` cache directory and stream it to device."""
        need = ("tokens.npy", "offsets.npy", "targets.npy")
        for name in need:
            full = os.path.join(path, name)
            if not os.path.exists(full):
                raise FileNotFoundError(f"missing {full}")
        tokens_np = np.load(os.path.join(path, "tokens.npy"), mmap_mode="r")
        offsets_np = np.load(os.path.join(path, "offsets.npy"), mmap_mode="r")
        targets_np = np.load(os.path.join(path, "targets.npy"), mmap_mode="r")

        meta: dict = {}
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

        tokens = _copy_to_device(tokens_np, device, torch.uint8)
        offsets = _copy_to_device(offsets_np, device, torch.int32)
        targets = _copy_to_device(targets_np, device, torch.float16, chunk_rows=1 << 18)
        return tokens, offsets, targets, meta

    @staticmethod
    def _read_safetensors(
        path: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Load a ``.safetensors`` cache directly onto device."""
        try:
            from safetensors.torch import load_file  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "safetensors is required to load a .safetensors corpus"
            ) from exc
        blob = load_file(path, device=str(device))
        tokens = blob["tokens"].to(torch.uint8)
        offsets = blob["offsets"].to(torch.int32)
        targets = blob["targets"].to(torch.float16)
        return tokens, offsets, targets, {}

    @classmethod
    def from_arrays(
        cls,
        tokens: np.ndarray,
        offsets: np.ndarray,
        targets: np.ndarray,
        device: str = "cpu",
        meta: Optional[dict] = None,
    ) -> "GpuCorpus":
        """Build a corpus from in-memory numpy arrays (tests, small caches).

        Args:
            tokens: uint8 ``[total_residues]``.
            offsets: integer ``[n_seqs + 1]``.
            targets: ``[total_residues, target_dim]``, cast to fp16.
            device: destination device.
            meta: optional metadata dict.

        Returns:
            A device-resident :class:`GpuCorpus`.
        """
        dev = torch.device(device)
        return cls(
            tokens=torch.as_tensor(np.asarray(tokens), dtype=torch.uint8, device=dev),
            offsets=torch.as_tensor(np.asarray(offsets), dtype=torch.int32, device=dev),
            targets=torch.as_tensor(
                np.asarray(targets), dtype=torch.float16, device=dev
            ),
            meta=meta,
        )

    # ------------------------------------------------------------- properties

    @property
    def device(self) -> torch.device:
        """Device every corpus tensor lives on."""
        return self.tokens.device

    @property
    def n_seqs(self) -> int:
        """Number of sequences in the corpus."""
        return int(self.lengths.shape[0])

    @property
    def total_residues(self) -> int:
        """Total number of residues across all sequences."""
        return int(self.tokens.shape[0])

    @property
    def target_dim(self) -> int:
        """Width of the target bank (128 in this project)."""
        return int(self.targets.shape[1])

    @property
    def nbytes(self) -> int:
        """Total device bytes held by the corpus tensors."""
        parts = (self.tokens, self.offsets, self.targets, self.lengths)
        return sum(t.numel() * t.element_size() for t in parts)

    def summary(self) -> dict:
        """Residency report for startup logging.

        Returns:
            A dict of per-tensor and total sizes plus length statistics.  Calls
            ``.item()``; startup only, never inside the training loop.
        """
        lengths_f = self.lengths.to(torch.float32)
        mb = 1024.0 * 1024.0

        def _entry(t: torch.Tensor) -> dict:
            return {
                "shape": tuple(int(s) for s in t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "mb": round(t.numel() * t.element_size() / mb, 3),
            }

        return {
            "device": str(self.device),
            "n_seqs": self.n_seqs,
            "total_residues": self.total_residues,
            "target_dim": self.target_dim,
            "tokens": _entry(self.tokens),
            "offsets": _entry(self.offsets),
            "targets": _entry(self.targets),
            "lengths": _entry(self.lengths),
            "total_mb": round(self.nbytes / mb, 3),
            "total_gb": round(self.nbytes / (mb * 1024.0), 4),
            "length_min": int(self.lengths.min()),
            "length_max": int(self.lengths.max()),
            "length_mean": float(lengths_f.mean()),
            "meta": self.meta,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GpuCorpus(n_seqs={self.n_seqs}, residues={self.total_residues}, "
            f"target_dim={self.target_dim}, device={self.device}, "
            f"{self.nbytes / 1024 ** 3:.3f} GiB)"
        )


class BucketBatcher:
    """Yields fixed-shape batches so ``torch.compile`` caches few graphs.

    Buckets are 128 / 256 / 384 / 512.  The per-bucket batch size is chosen so
    ``B * L`` stays close to ``token_budget`` (token-budget batching), which
    keeps padding waste under 8% instead of the ~48% a pad-to-512 collate would
    burn.  Token-budget batching alone is not sufficient for that: with
    pad-up-to-the-next-bucket assignment (``policy="pad"``) a Swiss-Prot-like
    length distribution still wastes ~17%.  The default ``policy="hybrid"``
    pads up only when the sequence nearly fills the bucket and otherwise crops
    down to the bucket below (fresh random crop window each draw), measuring
    ~96% padding efficiency on that distribution.

    All index math runs on device:

    * bucket assignment and the per-bucket index lists are computed **once** at
      construction with ``torch.bucketize`` / ``torch.nonzero``;
    * each epoch permutes each bucket with ``torch.randperm(device=...)``;
    * each batch gathers ``[B, L]`` tokens and ``[B, L, D]`` targets with a
      single advanced index using one broadcast index matrix.

    The only host-side state is the step counter and the (static) interleave
    pattern of bucket ids, so no step performs a device synchronisation.
    """

    def __init__(
        self,
        corpus: GpuCorpus,
        buckets: Sequence[int] = DEFAULT_BUCKETS,
        token_budget: int = TOKEN_BUDGET,
        masker: Optional["object"] = None,
        seed: int = 0,
        drop_last: bool = True,
        shuffle_bucket_order: bool = True,
        batch_multiple: int = 8,
        pad_id: int = PAD_ID,
        min_length: int = 1,
        policy: str = "hybrid",
        pad_threshold: float = 0.85,
    ) -> None:
        """Precompute bucket membership and per-bucket batch sizes.

        Args:
            corpus: the device-resident corpus.
            buckets: ascending bucket lengths.  Sequences longer than the last
                bucket are truncated to it.
            token_budget: target ``B * L`` tokens per step.
            masker: optional object with
                ``__call__(tokens, pad_mask) -> (tokens, mask_sel, labels)``
                (see :mod:`xjepa.data.masking`).  If ``None`` the batch carries
                an all-``False`` ``mask_sel`` and all-``-100`` ``labels``.
            seed: seed for the device RNG driving epoch permutations.
            drop_last: drop the trailing partial batch of each bucket so every
                emitted batch has a static shape.
            shuffle_bucket_order: shuffle the order in which buckets' batches
                are emitted.  Uses a host-side ``random.Random`` over small
                python ints, never a tensor, so it costs no device sync.
            batch_multiple: round per-bucket batch sizes down to this multiple.
            pad_id: token id written at padding positions.
            min_length: sequences shorter than this are dropped.
            policy: bucket assignment rule.

                * ``"hybrid"`` (default): pad up when the sequence nearly fills
                  the next bucket (occupancy ``>= pad_threshold``), otherwise
                  crop down.  Keeps padding waste low *and* discards far fewer
                  residues per epoch than pure cropping.
                * ``"crop"``: a sequence goes to the **largest**
                  bucket that is no longer than it, and is randomly cropped to
                  that length (a fresh crop offset every time it is drawn, so
                  the whole sequence is covered across epochs).  Only sequences
                  shorter than the smallest bucket ever pad, which is what
                  keeps padding waste under 8%.
                * ``"pad"``: a sequence goes to the **smallest** bucket that
                  fits it and is padded up.  Loses no residues but wastes far
                  more slots (~17% on a Swiss-Prot-like length distribution);
                  provided for ablation.
            pad_threshold: minimum occupancy ``len / bucket`` required to pad
                up rather than crop down, used by ``policy="hybrid"``.

        Raises:
            ValueError: if ``buckets`` is not ascending/positive, or ``policy``
                is unknown.
        """
        if policy not in ("crop", "pad", "hybrid"):
            raise ValueError(f"unknown bucket policy {policy!r}")
        if len(buckets) == 0 or any(b <= 0 for b in buckets):
            raise ValueError("buckets must be positive")
        if list(buckets) != sorted(buckets):
            raise ValueError("buckets must be ascending")

        self.corpus = corpus
        self.buckets: tuple[int, ...] = tuple(int(b) for b in buckets)
        self.token_budget = int(token_budget)
        self.masker = masker
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle_bucket_order = bool(shuffle_bucket_order)
        self.pad_id = int(pad_id)
        self.max_len = self.buckets[-1]
        self.policy = policy
        self.pad_threshold = float(pad_threshold)

        device = corpus.device
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)
        self._order_rng = random.Random(self.seed ^ 0x5EED)
        self._epoch = 0

        # --- per-bucket batch sizes: B * L ~ token_budget, static per bucket.
        self.batch_sizes: tuple[int, ...] = tuple(
            self._batch_size_for(b, batch_multiple) for b in self.buckets
        )

        # --- bucket assignment, computed ONCE on device.
        lengths = corpus.lengths.to(torch.int64)
        keep = lengths >= int(min_length)
        edges_all = torch.tensor(self.buckets, device=device, dtype=torch.int64)
        n_buckets = len(self.buckets)
        if policy in ("crop", "hybrid"):
            # number of buckets <= len, minus one -> largest bucket that fits
            # inside the sequence; shorter-than-smallest falls back to bucket 0.
            down = (torch.bucketize(lengths, edges_all, right=True) - 1).clamp_min(0)
            if policy == "crop":
                bucket_of = down
            else:
                # smallest bucket >= len (n_buckets if the sequence overflows).
                up = torch.bucketize(lengths, edges_all, right=False)
                up_c = up.clamp(max=n_buckets - 1)
                occupancy = lengths.to(torch.float32) / edges_all[up_c].to(
                    torch.float32
                )
                pad_up = (up < n_buckets) & (occupancy >= self.pad_threshold)
                bucket_of = torch.where(pad_up, up_c, down)
        else:
            edges = edges_all[:-1]
            eff = lengths.clamp(max=self.max_len)
            # right=True -> bucket i holds edges[i-1] < len <= edges[i].
            bucket_of = torch.bucketize(eff, edges, right=True)

        self._bucket_index: list[torch.Tensor] = []
        self._bucket_counts: list[int] = []
        self._n_batches: list[int] = []
        self._arange: list[torch.Tensor] = []
        for i, blen in enumerate(self.buckets):
            idx = torch.nonzero((bucket_of == i) & keep, as_tuple=False).flatten()
            self._bucket_index.append(idx.to(torch.int64))
            n = int(idx.numel())
            self._bucket_counts.append(n)
            bs = self.batch_sizes[i]
            self._n_batches.append(n // bs if self.drop_last else math.ceil(n / bs))
            self._arange.append(torch.arange(blen, device=device, dtype=torch.int64))

        # --- static interleave pattern of bucket ids (python ints only).
        self._pattern: list[int] = self._build_pattern()

    # ------------------------------------------------------------ construction

    def _batch_size_for(self, bucket_len: int, multiple: int) -> int:
        """Largest batch size at ``bucket_len`` within the token budget."""
        raw = max(1, self.token_budget // int(bucket_len))
        if multiple > 1 and raw >= multiple:
            raw = (raw // multiple) * multiple
        return int(raw)

    def _build_pattern(self) -> list[int]:
        """Proportional round-robin order of bucket ids for one epoch.

        The *content* of each batch is shuffled on device every epoch; only the
        order in which buckets take turns is host-side, so iterating never
        forces a device sync.
        """
        total = sum(self._n_batches)
        if total == 0:
            return []
        remaining = list(self._n_batches)
        pattern: list[int] = []
        # Largest-remainder round robin: repeatedly emit the bucket with the
        # most batches left, which spreads each bucket evenly over the epoch.
        while len(pattern) < total:
            i = max(range(len(remaining)), key=lambda k: remaining[k])
            pattern.append(i)
            remaining[i] -= 1
        return pattern

    # --------------------------------------------------------------- iteration

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return sum(self._n_batches)

    @property
    def steps_per_epoch(self) -> int:
        """Alias of ``len(self)`` for trainer readability."""
        return len(self)

    def bucket_counts(self) -> dict[int, int]:
        """Sequences assigned to each bucket length."""
        return {b: c for b, c in zip(self.buckets, self._bucket_counts)}

    def batch_shapes(self) -> dict[int, tuple[int, int]]:
        """The static ``(B, L)`` shape emitted for each bucket length."""
        return {b: (bs, b) for b, bs in zip(self.buckets, self.batch_sizes)}

    def padding_efficiency(self) -> float:
        """Fraction of emitted ``B * L`` slots that hold a real residue.

        Computed from the static bucket assignment (and the number of full
        batches each bucket yields), so it is exact in expectation over epoch
        permutations and independent of any particular epoch.

        Returns:
            Real tokens / padded tokens in ``[0, 1]``.  Must exceed 0.92.
        """
        real = 0.0
        padded = 0.0
        for i, blen in enumerate(self.buckets):
            n = self._bucket_counts[i]
            if n == 0:
                continue
            used = min(n, self._n_batches[i] * self.batch_sizes[i])
            if used == 0:
                continue
            idx = self._bucket_index[i]
            lens = self.corpus.lengths[idx].to(torch.float64).clamp(max=float(blen))
            mean_len = float(lens.mean())
            real += mean_len * used
            padded += float(blen) * used
        if padded == 0.0:
            return 0.0
        return real / padded

    def _permutations(self) -> list[torch.Tensor]:
        """Fresh on-device permutation of every bucket's index list."""
        out = []
        for idx in self._bucket_index:
            n = int(idx.numel())
            if n == 0:
                out.append(idx)
                continue
            perm = torch.randperm(n, generator=self.generator, device=self.device)
            out.append(idx[perm])
        return out

    def make_batch(self, bucket_i: int, rows: torch.Tensor) -> Batch:
        """Gather one padded batch for ``rows`` (sequence ids) in one shot.

        Builds ``index[b, t] = offsets[rows[b]] + t`` by broadcasting
        ``torch.arange(L)`` against the per-sequence start offsets, clamps the
        out-of-sequence entries to a safe slot, then uses that single index
        matrix twice: once to gather ``[B, L]`` tokens and once to gather
        ``[B, L, D]`` targets.  No python loop over sequences.

        Args:
            bucket_i: index into ``self.buckets``.
            rows: int64 device tensor ``[B]`` of sequence ids.

        Returns:
            A :class:`Batch` on device.
        """
        blen = self.buckets[bucket_i]
        ar = self._arange[bucket_i]  # [L]
        starts = self.corpus.offsets[rows].to(torch.int64)  # [B]
        raw_lens = self.corpus.lengths[rows].to(torch.int64)  # [B]
        lens = raw_lens.clamp(max=blen)  # [B]
        if self.policy != "pad":
            # Random crop window for sequences longer than the bucket; drawn on
            # device, so the whole sequence is covered across epochs.
            slack = (raw_lens - blen).clamp_min(0)  # [B]
            u = torch.rand(
                slack.shape, generator=self.generator, device=slack.device
            )
            starts = starts + (u * (slack + 1).to(u.dtype)).to(torch.int64).clamp(
                max=slack
            )

        pad_mask = ar.unsqueeze(0) < lens.unsqueeze(1)  # [B, L] True = real
        flat = starts.unsqueeze(1) + ar.unsqueeze(0)  # [B, L]
        # Clamp padding slots onto the sequence's first residue: always in
        # range, and their gathered values are overwritten/zeroed below.
        flat = torch.where(pad_mask, flat, starts.unsqueeze(1))

        tokens = self.corpus.tokens[flat].to(torch.int64)  # [B, L]
        tokens = torch.where(
            pad_mask, tokens, torch.full_like(tokens, self.pad_id)
        )
        targets = self.corpus.targets[flat]  # [B, L, D]
        targets = targets * pad_mask.unsqueeze(-1).to(targets.dtype)

        if self.masker is not None:
            tokens, mask_sel, labels = self.masker(tokens, pad_mask)
        else:
            mask_sel = torch.zeros_like(pad_mask)
            labels = torch.full_like(tokens, -100)

        return Batch(
            tokens=tokens,
            targets=targets,
            pad_mask=pad_mask,
            mask_sel=mask_sel,
            labels=labels,
            bucket=blen,
        )

    def __iter__(self) -> Iterator[Batch]:
        """Iterate one epoch of fixed-shape batches.

        Yields:
            :class:`Batch` objects; shapes cycle over the (small) set
            ``{(B_b, L_b)}``, so ``torch.compile`` sees at most
            ``len(buckets)`` graphs.
        """
        perms = self._permutations()
        cursors = [0] * len(self.buckets)
        pattern = list(self._pattern)
        if self.shuffle_bucket_order:
            self._order_rng.shuffle(pattern)
        self._epoch += 1
        for bucket_i in pattern:
            bs = self.batch_sizes[bucket_i]
            start = cursors[bucket_i]
            stop = min(start + bs, int(perms[bucket_i].numel()))
            cursors[bucket_i] = stop
            rows = perms[bucket_i][start:stop]
            if rows.numel() == 0:
                continue
            if self.drop_last and int(rows.numel()) < bs:
                continue
            yield self.make_batch(bucket_i, rows)

    def summary(self) -> dict:
        """Static batching plan, for startup logging."""
        return {
            "buckets": list(self.buckets),
            "batch_sizes": list(self.batch_sizes),
            "tokens_per_step": [b * s for b, s in zip(self.buckets, self.batch_sizes)],
            "token_budget": self.token_budget,
            "bucket_counts": self.bucket_counts(),
            "batches_per_bucket": list(self._n_batches),
            "steps_per_epoch": len(self),
            "policy": self.policy,
            "pad_threshold": self.pad_threshold,
            "padding_efficiency": round(self.padding_efficiency(), 5),
            "crop_loss_fraction": round(self.crop_loss_fraction(), 5),
        }

    def crop_loss_fraction(self) -> float:
        """Fraction of a sequence's residues skipped by cropping, per epoch.

        Cropping does not waste compute (unlike padding) and the skipped window
        moves every epoch, but the number belongs next to
        :meth:`padding_efficiency` so the two trade-offs are reported together.

        Returns:
            Dropped residues / total residues of the used sequences.
        """
        dropped = 0.0
        total = 0.0
        for i, blen in enumerate(self.buckets):
            n = self._bucket_counts[i]
            if n == 0:
                continue
            used = min(n, self._n_batches[i] * self.batch_sizes[i])
            if used == 0:
                continue
            lens = self.corpus.lengths[self._bucket_index[i]].to(torch.float64)
            frac = used / float(n)
            total += float(lens.sum()) * frac
            dropped += float((lens - float(blen)).clamp_min(0.0).sum()) * frac
        if total == 0.0:
            return 0.0
        return dropped / total
