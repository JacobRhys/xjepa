"""On-device masking for the MLM and JEPA objectives.

Every random number is drawn from a ``torch.Generator(device=...)`` so nothing
is sampled on the host and nothing is transferred per step.  Two schemes are
provided:

* **random** — i.i.d. Bernoulli masking at ``rate`` (ESM-2 style, 15%).
* **span** — geometric span lengths with mean ``mean_span`` (default 8),
  SpanBERT style, vectorised with a ``cummax`` reach trick (no python loop).

Corruption follows ESM-2's 80/10/10 rule on the MLM path; on the JEPA paths a
masked position simply becomes the mask token.  Padding is **never** masked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["MaskingConfig", "Masker", "random_mask", "span_mask", "apply_corruption"]

PAD_ID: int = 1
MASK_ID: int = 32
# Inclusive id range of the 20 standard amino acids in the ESM-2 alphabet.
AA_LO: int = 4
AA_HI: int = 23
IGNORE_INDEX: int = -100


@dataclass
class MaskingConfig:
    """Masking hyper-parameters.

    Attributes:
        mode: ``"random"`` or ``"span"``.
        rate: fraction of *real* residues to mask (0.15).
        mean_span: mean geometric span length for ``mode="span"``.
        max_span: hard cap on a single span's length.
        corruption: ``"mlm"`` for ESM-2 80/10/10, ``"jepa"`` to replace every
            masked position with the mask token, ``"none"`` to leave the input
            tokens untouched (mask is then only a loss selector).
        mask_id: id of ``<mask>``.
        pad_id: id of ``<pad>``.
        rand_lo: lowest id used for the 10% random-replacement branch.
        rand_hi: highest id (inclusive) used for random replacement.
        ensure_one: guarantee at least one masked position per sequence.
    """

    mode: str = "random"
    rate: float = 0.15
    mean_span: float = 8.0
    max_span: int = 32
    corruption: str = "mlm"
    mask_id: int = MASK_ID
    pad_id: int = PAD_ID
    rand_lo: int = AA_LO
    rand_hi: int = AA_HI
    ensure_one: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("random", "span"):
            raise ValueError(f"unknown masking mode {self.mode!r}")
        if self.corruption not in ("mlm", "jepa", "none"):
            raise ValueError(f"unknown corruption {self.corruption!r}")
        if not 0.0 < self.rate < 1.0:
            raise ValueError("rate must be in (0, 1)")
        if self.mean_span < 1.0:
            raise ValueError("mean_span must be >= 1")


def _ensure_one(
    mask_sel: torch.Tensor, pad_mask: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """Force one masked position in rows that drew none.

    Branch-free (no ``if tensor.any()``, which would sync the device): the
    fallback position is the real residue with the smallest ``scores`` value.

    Args:
        mask_sel: bool ``[B, L]`` current selection.
        pad_mask: bool ``[B, L]``, ``True`` = real residue.
        scores: float ``[B, L]`` tiebreak scores (uniform noise).

    Returns:
        bool ``[B, L]`` selection with every row non-empty (where the row has
        at least one real residue).
    """
    empty = ~mask_sel.any(dim=1, keepdim=True)  # [B, 1]
    cand = scores.masked_fill(~pad_mask, float("inf")).argmin(dim=1)  # [B]
    onehot = torch.zeros_like(mask_sel)
    onehot.scatter_(1, cand.unsqueeze(1), True)
    return mask_sel | (empty & onehot & pad_mask)


def random_mask(
    pad_mask: torch.Tensor,
    rate: float,
    generator: torch.Generator,
    ensure_one: bool = True,
) -> torch.Tensor:
    """I.i.d. Bernoulli masking over real residues only.

    Args:
        pad_mask: bool ``[B, L]``, ``True`` = real residue.
        rate: masking probability.
        generator: device generator.
        ensure_one: guarantee a non-empty selection per row.

    Returns:
        bool ``[B, L]`` mask selection, always a subset of ``pad_mask``.
    """
    noise = torch.rand(
        pad_mask.shape, generator=generator, device=pad_mask.device, dtype=torch.float32
    )
    mask_sel = (noise < rate) & pad_mask
    if ensure_one:
        mask_sel = _ensure_one(mask_sel, pad_mask, noise)
    return mask_sel


def span_mask(
    pad_mask: torch.Tensor,
    rate: float,
    mean_span: float,
    generator: torch.Generator,
    max_span: int = 32,
    ensure_one: bool = True,
) -> torch.Tensor:
    """Geometric-length span masking, fully vectorised on device.

    Span starts are Bernoulli with probability ``p`` chosen so the expected
    coverage of the (overlapping) span process equals ``rate``: a position is
    left uncovered with probability ``prod_k (1 - p (1-q)^k) ~ exp(-p/q)``
    with ``q = 1 / mean_span``, hence ``p = -ln(1 - rate) / mean_span``.

    Each start at position ``i`` with sampled length ``l`` covers
    ``[i, i + l)``.  Coverage is materialised without a loop by writing
    ``i + l`` into a "reach" array at start positions and taking a running
    ``cummax`` along the sequence: position ``j`` is covered iff
    ``j < cummax(reach)[j]``.

    Args:
        pad_mask: bool ``[B, L]``, ``True`` = real residue.
        rate: target fraction of real residues masked.
        mean_span: mean span length (geometric, support ``>= 1``).
        generator: device generator.
        max_span: cap on a single span length.
        ensure_one: guarantee a non-empty selection per row.

    Returns:
        bool ``[B, L]`` mask selection, always a subset of ``pad_mask``.
    """
    device = pad_mask.device
    b, length = pad_mask.shape
    q = 1.0 / float(mean_span)
    p_start = -math.log1p(-float(rate)) / float(mean_span)

    u_start = torch.rand(
        (b, length), generator=generator, device=device, dtype=torch.float32
    )
    u_len = torch.rand(
        (b, length), generator=generator, device=device, dtype=torch.float32
    )
    # Geometric on {1, 2, ...} with mean 1/q.
    lens = torch.ceil(torch.log(u_len.clamp_min(1e-9)) / math.log1p(-q))
    lens = lens.clamp(1.0, float(max_span)).to(torch.int64)

    starts = (u_start < p_start) & pad_mask
    ar = torch.arange(length, device=device, dtype=torch.int64).unsqueeze(0)
    reach = torch.where(starts, ar + lens, torch.zeros_like(lens))
    reach = torch.cummax(reach, dim=1).values
    mask_sel = (ar < reach) & pad_mask
    if ensure_one:
        mask_sel = _ensure_one(mask_sel, pad_mask, u_start)
    return mask_sel


def apply_corruption(
    tokens: torch.Tensor,
    mask_sel: torch.Tensor,
    cfg: MaskingConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace masked tokens per ``cfg.corruption``.

    ``"mlm"`` implements ESM-2's 80/10/10 split (mask / random amino acid /
    unchanged); ``"jepa"`` sends every masked position to ``<mask>``;
    ``"none"`` leaves the tokens untouched.

    Args:
        tokens: int64 ``[B, L]`` original token ids.
        mask_sel: bool ``[B, L]`` masked positions.
        cfg: masking configuration.
        generator: device generator.

    Returns:
        int64 ``[B, L]`` corrupted tokens (a new tensor).
    """
    if cfg.corruption == "none":
        return tokens.clone()
    if cfg.corruption == "jepa":
        return torch.where(
            mask_sel, torch.full_like(tokens, cfg.mask_id), tokens
        )
    # "mlm": 80% <mask>, 10% random amino acid, 10% unchanged.
    u = torch.rand(
        tokens.shape, generator=generator, device=tokens.device, dtype=torch.float32
    )
    rand_tok = torch.randint(
        low=cfg.rand_lo,
        high=cfg.rand_hi + 1,
        size=tokens.shape,
        generator=generator,
        device=tokens.device,
        dtype=tokens.dtype,
    )
    out = torch.where(
        mask_sel & (u < 0.8), torch.full_like(tokens, cfg.mask_id), tokens
    )
    out = torch.where(mask_sel & (u >= 0.8) & (u < 0.9), rand_tok, out)
    return out


class Masker:
    """Callable that masks a batch in place of a collate function.

    Example:
        >>> cfg = MaskingConfig(mode="span", corruption="jepa")
        >>> masker = Masker(cfg, device="cpu", seed=0)
        >>> tokens, mask_sel, labels = masker(tokens, pad_mask)
    """

    def __init__(
        self,
        cfg: MaskingConfig,
        device: str | torch.device = "cpu",
        seed: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Create a masker bound to one device generator.

        Args:
            cfg: masking configuration.
            device: device the batches live on.
            seed: RNG seed (ignored if ``generator`` is given).
            generator: an existing device generator to share.
        """
        self.cfg = cfg
        self.device = torch.device(device)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        self.generator = generator

    def select(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """Draw a mask selection for ``pad_mask`` without corrupting tokens."""
        if self.cfg.mode == "random":
            return random_mask(
                pad_mask, self.cfg.rate, self.generator, self.cfg.ensure_one
            )
        return span_mask(
            pad_mask,
            self.cfg.rate,
            self.cfg.mean_span,
            self.generator,
            self.cfg.max_span,
            self.cfg.ensure_one,
        )

    def __call__(
        self, tokens: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask and corrupt one batch.

        Args:
            tokens: int64 ``[B, L]`` clean token ids.
            pad_mask: bool ``[B, L]``, ``True`` = real residue.

        Returns:
            ``(tokens_out, mask_sel, labels)`` where ``tokens_out`` is the
            corrupted model input, ``mask_sel`` is the bool ``[B, L]`` masked
            selection (never overlapping padding) and ``labels`` holds the
            original id at masked positions and ``-100`` elsewhere.
        """
        mask_sel = self.select(pad_mask)
        labels = torch.where(
            mask_sel, tokens, torch.full_like(tokens, IGNORE_INDEX)
        )
        tokens_out = apply_corruption(tokens, mask_sel, self.cfg, self.generator)
        return tokens_out, mask_sel, labels
