"""Frozen-feature probes.

The experimental control (RESEARCH_PLAN.md 5.2) is that *one* hyperparameter grid
per task is applied identically to every condition -- never tuned per condition.
That is enforced structurally here: the grids are module-level constants
(:data:`LINEAR_GRID`, :data:`MLP_GRID`), the task runners take a grid argument
that defaults to them, and selection is always on the validation split with the
test score reported for the val-selected config only.

Probes provided
---------------
* per-residue classification (SS3 / SS8) -- accuracy;
* pairwise contact prediction with a bilinear probe, 20k sampled pairs per
  protein during training -- P@L/5 for medium (|i-j| in [12, 24)) and long
  (|i-j| >= 24) range;
* mean-pooled scalar regression (fluorescence, stability) -- Spearman rho.

Each comes in a linear and a 1-hidden-layer-256 MLP variant; linear is primary.

Features are extracted ONCE per checkpoint (:func:`extract_features`) and cached
to disk (:class:`FeatureCache`) so that the whole probe sweep is cheap and never
re-runs the encoder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ProbeConfig",
    "LINEAR_GRID",
    "MLP_GRID",
    "DEFAULT_GRID",
    "ProbeResult",
    "spearman_rho",
    "LinearProbe",
    "MlpProbe",
    "BilinearContactProbe",
    "extract_features",
    "FeatureCache",
    "ResidueTaskData",
    "ContactExample",
    "PooledTaskData",
    "run_residue_classification_probe",
    "run_contact_probe",
    "run_regression_probe",
]


# --------------------------------------------------------------------------------------
# Hyperparameter grid -- identical for every condition. This is the control.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """One point of the probe hyperparameter grid.

    Args:
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (this is the ridge strength for the
            linear probes).
        epochs: passes over the probe training set.
        batch_size: probe minibatch size (rows for residue/pooled tasks,
            proteins for the contact task).
        hidden: ``None`` for a linear probe, otherwise the hidden width (256 for
            the MLP variant mandated by the plan).
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 1024
    hidden: Optional[int] = None

    @property
    def kind(self) -> str:
        return "linear" if self.hidden is None else f"mlp{self.hidden}"

    def tag(self) -> str:
        return f"{self.kind}|lr{self.lr:g}|wd{self.weight_decay:g}|ep{self.epochs}"


#: Primary probe family: linear head, three regularisation strengths.
LINEAR_GRID: Tuple[ProbeConfig, ...] = (
    ProbeConfig(lr=1e-3, weight_decay=1e-5, epochs=20, hidden=None),
    ProbeConfig(lr=1e-3, weight_decay=1e-3, epochs=20, hidden=None),
    ProbeConfig(lr=3e-3, weight_decay=1e-4, epochs=20, hidden=None),
)

#: Secondary probe family: one hidden layer of 256 units.
MLP_GRID: Tuple[ProbeConfig, ...] = (
    ProbeConfig(lr=1e-3, weight_decay=1e-5, epochs=20, hidden=256),
    ProbeConfig(lr=1e-3, weight_decay=1e-3, epochs=20, hidden=256),
    ProbeConfig(lr=3e-3, weight_decay=1e-4, epochs=20, hidden=256),
)

#: The full sweep run for every task and every condition.
DEFAULT_GRID: Tuple[ProbeConfig, ...] = LINEAR_GRID + MLP_GRID


@dataclass
class ProbeResult:
    """Outcome of one probe sweep over one task for one checkpoint."""

    task: str
    best_config: ProbeConfig
    val_metric: float
    test_metrics: Dict[str, float]
    all_val_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "best_config": asdict(self.best_config),
            "val_metric": self.val_metric,
            "test_metrics": dict(self.test_metrics),
            "all_val_metrics": dict(self.all_val_metrics),
        }


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (1-based), ties averaged -- the scipy ``rankdata`` default."""
    x = x.reshape(-1).double()
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
    sx = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def spearman_rho(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman rank correlation, ties averaged.

    Args:
        pred: predicted scores, any shape (flattened).
        target: ground-truth values, same number of elements.

    Returns:
        rho in ``[-1, 1]``; 0.0 when either side is constant.
    """
    pred = pred.detach().reshape(-1).cpu()
    target = target.detach().reshape(-1).cpu()
    if pred.numel() != target.numel():
        raise ValueError("spearman_rho: length mismatch")
    if pred.numel() < 2:
        return 0.0
    rp, rt = _rankdata(pred), _rankdata(target)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = rp.norm() * rt.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((rp @ rt) / denom)


# --------------------------------------------------------------------------------------
# Probe heads
# --------------------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """A single affine map ``[*, in_dim] -> [*, out_dim]``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MlpProbe(nn.Module):
    """One hidden layer of ``hidden`` units with ReLU, then a linear output."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_head(in_dim: int, out_dim: int, cfg: ProbeConfig) -> nn.Module:
    if cfg.hidden is None:
        return LinearProbe(in_dim, out_dim)
    return MlpProbe(in_dim, out_dim, hidden=cfg.hidden)


class BilinearContactProbe(nn.Module):
    """Low-rank bilinear scorer over residue pairs.

    ``score(i, j) = <A(h_i), B(h_j)> + <A(h_j), B(h_i)>) / 2 + bias``

    The symmetrisation is explicit so the probe cannot cheat on the i<j
    convention of the contact map. ``A`` and ``B`` are the probe family selected
    by :class:`ProbeConfig` (linear, or 256-unit MLP), projecting to ``rank``.

    Args:
        in_dim: residue feature dimension.
        cfg: probe config (controls linear vs MLP and is what the grid varies).
        rank: bilinear rank; 64 keeps the probe small relative to the encoder.
    """

    def __init__(self, in_dim: int, cfg: ProbeConfig, rank: int = 64) -> None:
        super().__init__()
        self.left = _build_head(in_dim, rank, cfg)
        self.right = _build_head(in_dim, rank, cfg)
        self.bias = nn.Parameter(torch.zeros(()))
        self.rank = rank

    def pair_logits(
        self, feats: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor
    ) -> torch.Tensor:
        """Score a list of pairs.

        Args:
            feats: ``[L, D]`` residue features for one protein.
            idx_i: ``[P]`` int64 row indices.
            idx_j: ``[P]`` int64 column indices.

        Returns:
            ``[P]`` logits.
        """
        a = self.left(feats)
        b = self.right(feats)
        fwd = (a[idx_i] * b[idx_j]).sum(-1)
        bwd = (a[idx_j] * b[idx_i]).sum(-1)
        return 0.5 * (fwd + bwd) + self.bias

    def full_map(self, feats: torch.Tensor) -> torch.Tensor:
        """Score every pair: ``[L, D] -> [L, L]`` symmetric logits."""
        a = self.left(feats)
        b = self.right(feats)
        s = a @ b.transpose(0, 1)
        return 0.5 * (s + s.transpose(0, 1)) + self.bias

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.full_map(feats)


# --------------------------------------------------------------------------------------
# Feature extraction + on-disk cache
# --------------------------------------------------------------------------------------


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    tokens: torch.Tensor,
    pad_mask: torch.Tensor,
    batch_size: int = 64,
    pool: str = "residue",
    device: Optional[torch.device] = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run the frozen encoder once over a dataset.

    Args:
        encoder: anything with ``forward(tokens, pad_mask) -> [B, L, D]`` (the
            ``xjepa/model/encoder.py`` contract).
        tokens: ``[N, L]`` int64 token ids (CPU or device).
        pad_mask: ``[N, L]`` bool, True = real residue.
        batch_size: rows per forward pass.
        pool: ``"residue"`` keeps ``[N, L, D]``; ``"mean"`` returns masked mean
            pooled ``[N, D]``.
        device: device to run on; defaults to the encoder's parameter device.
        out_dtype: storage dtype for the cache (fp16 halves the cache size and
            is well inside probe noise).

    Returns:
        ``[N, L, D]`` or ``[N, D]`` on CPU, in ``out_dtype``.
    """
    if pool not in ("residue", "mean"):
        raise ValueError("pool must be 'residue' or 'mean'")
    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:  # parameter-less stub encoders (tests)
            device = tokens.device
    was_training = encoder.training
    encoder.eval()
    chunks: List[torch.Tensor] = []
    try:
        for start in range(0, tokens.shape[0], batch_size):
            tk = tokens[start : start + batch_size].to(device, non_blocking=True)
            pm = pad_mask[start : start + batch_size].to(device, non_blocking=True)
            h = encoder(tk, pm)
            if pool == "mean":
                h = mean_pool(h, pm)
            chunks.append(h.to(out_dtype).cpu())
    finally:
        encoder.train(was_training)
    return torch.cat(chunks, dim=0)


def mean_pool(features: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the length axis: ``[B, L, D] x [B, L] -> [B, D]``."""
    m = pad_mask.unsqueeze(-1).to(features.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (features * m).sum(dim=1) / denom


class FeatureCache:
    """Disk cache of extracted features, keyed by checkpoint id + dataset id.

    One directory per run; files are ``<checkpoint>__<dataset>__<pool>.pt``.
    Extraction happens at most once per (checkpoint, dataset, pool) triple, so the
    probe sweep -- which touches the same features 6+ times per task -- pays for
    the encoder exactly once.

    Args:
        root: cache directory; created on demand.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_id: str, dataset_id: str, pool: str) -> Path:
        key = f"{checkpoint_id}__{dataset_id}__{pool}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        safe = "".join(c if c.isalnum() or c in "-._" else "-" for c in key)[:80]
        return self.root / f"{safe}.{digest}.pt"

    def get_or_extract(
        self,
        checkpoint_id: str,
        dataset_id: str,
        extract_fn: Callable[[], torch.Tensor],
        pool: str = "residue",
    ) -> torch.Tensor:
        """Return cached features, calling ``extract_fn`` only on a miss."""
        path = self._path(checkpoint_id, dataset_id, pool)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        feats = extract_fn()
        tmp = path.with_suffix(".tmp")
        torch.save(feats, tmp)
        tmp.replace(path)
        return feats

    def write_manifest(self, name: str, payload: Dict[str, object]) -> Path:
        """Record what was extracted (shapes, dtypes, ids) next to the tensors."""
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path


# --------------------------------------------------------------------------------------
# Generic supervised probe fitting
# --------------------------------------------------------------------------------------


def _iter_minibatches(
    n: int, batch_size: int, generator: torch.Generator, shuffle: bool = True
) -> Iterable[torch.Tensor]:
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def _fit_head(
    head: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: ProbeConfig,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    seed: int = 0,
) -> nn.Module:
    """Fit a probe head on in-memory features with AdamW. Returns ``head``."""
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    x = x.to(device)
    y = y.to(device)
    head.train()
    for _ in range(cfg.epochs):
        for batch_idx in _iter_minibatches(x.shape[0], cfg.batch_size, gen):
            bi = batch_idx.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(x.index_select(0, bi)), y.index_select(0, bi))
            loss.backward()
            opt.step()
    head.eval()
    return head


# --------------------------------------------------------------------------------------
# Task 1: per-residue classification (SS3 / SS8)
# --------------------------------------------------------------------------------------


@dataclass
class ResidueTaskData:
    """Flattened per-residue features and integer labels for one split set.

    Attributes hold ``[N, D]`` float features and ``[N]`` int64 labels with the
    ignore positions already dropped (see :meth:`from_padded`).
    """

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int

    @staticmethod
    def _flatten(
        feats: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f = feats.reshape(-1, feats.shape[-1]).float()
        l = labels.reshape(-1).long()
        keep = l != ignore_index
        return f[keep], l[keep]

    @classmethod
    def from_padded(
        cls,
        splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        num_classes: int,
        ignore_index: int = -100,
    ) -> "ResidueTaskData":
        """Build from ``{"train"/"val"/"test": ([N, L, D] feats, [N, L] labels)}``.

        Labels equal to ``ignore_index`` (padding, unresolved residues) are
        dropped rather than masked, which makes the probe loop a plain
        classification problem over rows.
        """
        need = ("train", "val", "test")
        missing = [k for k in need if k not in splits]
        if missing:
            raise KeyError(f"missing splits: {missing}")
        flat = {k: cls._flatten(*splits[k], ignore_index=ignore_index) for k in need}
        return cls(
            train_x=flat["train"][0],
            train_y=flat["train"][1],
            val_x=flat["val"][0],
            val_y=flat["val"][1],
            test_x=flat["test"][0],
            test_y=flat["test"][1],
            num_classes=num_classes,
        )


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())


def run_residue_classification_probe(
    data: ResidueTaskData,
    task: str = "ss3",
    grid: Sequence[ProbeConfig] = DEFAULT_GRID,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> ProbeResult:
    """Sweep the fixed grid for an SS3/SS8-style per-residue classifier.

    Args:
        data: flattened features/labels for train/val/test.
        task: name recorded in the result (``"ss3"`` / ``"ss8"``).
        grid: hyperparameter grid; defaults to the shared
            :data:`DEFAULT_GRID`. Pass the *same* grid for every condition.
        device: torch device; defaults to CPU.
        seed: probe init/shuffle seed.

    Returns:
        :class:`ProbeResult` with ``test_metrics = {"accuracy": ...}`` for the
        val-selected config, plus the val accuracy of every grid point.
    """
    device = device or torch.device("cpu")
    in_dim = data.train_x.shape[-1]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy
    best: Optional[Tuple[float, ProbeConfig, nn.Module]] = None
    all_val: Dict[str, float] = {}
    for i, cfg in enumerate(grid):
        torch.manual_seed(seed + i)
        head = _build_head(in_dim, data.num_classes, cfg)
        head = _fit_head(head, data.train_x, data.train_y, cfg, loss_fn, device, seed=seed + i)
        with torch.no_grad():
            val_acc = _accuracy(head(data.val_x.to(device)), data.val_y.to(device))
        all_val[cfg.tag()] = val_acc
        if best is None or val_acc > best[0]:
            best = (val_acc, cfg, head)
    assert best is not None, "empty probe grid"
    val_acc, cfg, head = best
    with torch.no_grad():
        test_acc = _accuracy(head(data.test_x.to(device)), data.test_y.to(device))
    return ProbeResult(
        task=task,
        best_config=cfg,
        val_metric=val_acc,
        test_metrics={"accuracy": test_acc},
        all_val_metrics=all_val,
    )


# --------------------------------------------------------------------------------------
# Task 2: contact prediction
# --------------------------------------------------------------------------------------

MEDIUM_RANGE: Tuple[int, int] = (12, 24)  # |i - j| in [12, 24)
LONG_RANGE_MIN: int = 24


@dataclass
class ContactExample:
    """One protein for the contact probe.

    Args:
        features: ``[L, D]`` residue features (fp16 from the cache is fine).
        contacts: ``[L, L]`` bool/0-1, True = C-beta distance < 8 A.
        valid: ``[L, L]`` bool, False where either residue is unresolved. Defaults
            to all-True.
    """

    features: torch.Tensor
    contacts: torch.Tensor
    valid: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        L = self.features.shape[0]
        if self.contacts.shape != (L, L):
            raise ValueError("contacts must be [L, L] matching features")
        if self.valid is None:
            self.valid = torch.ones(L, L, dtype=torch.bool)

    @property
    def length(self) -> int:
        return int(self.features.shape[0])


def _range_mask(length: int, band: str, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask selecting a sequence-separation band."""
    i = torch.arange(length, device=device).unsqueeze(1)
    j = torch.arange(length, device=device).unsqueeze(0)
    sep = (j - i).abs()
    upper = j > i
    if band == "medium":
        return upper & (sep >= MEDIUM_RANGE[0]) & (sep < MEDIUM_RANGE[1])
    if band == "long":
        return upper & (sep >= LONG_RANGE_MIN)
    if band == "medium_long":
        return upper & (sep >= MEDIUM_RANGE[0])
    raise ValueError(f"unknown band {band!r}")


def _sample_pairs(
    example: ContactExample, n_pairs: int, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subsample at most ``n_pairs`` valid medium+long pairs from one protein.

    Returns ``(idx_i, idx_j, labels_float)``. Sampling is what keeps the contact
    probe affordable: a 400-residue protein has 80k upper-triangle pairs and the
    plan budgets 20k per protein.
    """
    device = example.features.device
    band = _range_mask(example.length, "medium_long", device) & example.valid.to(device)
    idx = band.nonzero(as_tuple=False)
    n = idx.shape[0]
    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, torch.zeros(0, device=device)
    if n > n_pairs:
        sel = torch.randperm(n, generator=generator)[:n_pairs].to(device)
        idx = idx.index_select(0, sel)
    ii, jj = idx[:, 0], idx[:, 1]
    y = example.contacts.to(device)[ii, jj].float()
    return ii, jj, y


def precision_at_l_over_k(
    logits: torch.Tensor,
    example: ContactExample,
    band: str = "long",
    k: int = 5,
) -> Optional[float]:
    """P@L/k for one protein in one separation band.

    Args:
        logits: ``[L, L]`` predicted scores.
        example: the protein (for contacts, validity and length).
        band: ``"medium"`` or ``"long"``.
        k: the divisor; 5 gives the standard P@L/5.

    Returns:
        Precision, or ``None`` when the band has no valid pairs for this protein
        (such proteins are skipped in the average rather than scored 0).
    """
    device = logits.device
    mask = _range_mask(example.length, band, device) & example.valid.to(device)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return None
    top_n = max(1, example.length // k)
    top_n = min(top_n, n_valid)
    scores = logits.masked_fill(~mask, float("-inf")).reshape(-1)
    order = torch.topk(scores, top_n).indices
    truth = example.contacts.to(device).reshape(-1).float()
    return float(truth[order].mean())


def _contact_eval(
    probe: BilinearContactProbe, examples: Sequence[ContactExample], device: torch.device
) -> Dict[str, float]:
    probe.eval()
    sums = {"medium": 0.0, "long": 0.0}
    counts = {"medium": 0, "long": 0}
    with torch.no_grad():
        for ex in examples:
            feats = ex.features.float().to(device)
            logits = probe.full_map(feats)
            for band in ("medium", "long"):
                p = precision_at_l_over_k(logits, ex, band=band, k=5)
                if p is not None:
                    sums[band] += p
                    counts[band] += 1
    # A band with no valid pairs in any protein scores NaN, not 0.0 -- reporting
    # zero would silently claim the probe failed when in fact nothing was scored
    # (e.g. every protein shorter than the long-range separation threshold).
    out = {
        f"p_at_l5_{band}": (sums[band] / counts[band] if counts[band] else float("nan"))
        for band in ("medium", "long")
    }
    scored = [out[f"p_at_l5_{b}"] for b in ("medium", "long") if counts[b]]
    out["p_at_l5_mean"] = sum(scored) / len(scored) if scored else float("nan")
    return out


def run_contact_probe(
    train: Sequence[ContactExample],
    val: Sequence[ContactExample],
    test: Sequence[ContactExample],
    grid: Sequence[ProbeConfig] = DEFAULT_GRID,
    pairs_per_protein: int = 20_000,
    rank: int = 64,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> ProbeResult:
    """Train a bilinear contact probe over the fixed grid; report P@L/5.

    Training subsamples ``pairs_per_protein`` medium+long-range pairs per protein
    per epoch (resampled each epoch, so over 20 epochs the probe still sees a
    broad slice of the map without ever materialising one).

    Args:
        train/val/test: protein-level examples.
        grid: shared hyperparameter grid.
        pairs_per_protein: sampling budget; the plan specifies 20k.
        rank: bilinear rank.
        device: torch device.
        seed: probe seed.

    Returns:
        :class:`ProbeResult` with medium/long/mean P@L/5 on test, selected on the
        mean P@L/5 of the validation split.
    """
    device = device or torch.device("cpu")
    in_dim = train[0].features.shape[-1]
    best: Optional[Tuple[float, ProbeConfig, BilinearContactProbe]] = None
    all_val: Dict[str, float] = {}
    for gi, cfg in enumerate(grid):
        torch.manual_seed(seed + gi)
        probe = BilinearContactProbe(in_dim, cfg, rank=rank).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(seed + gi)
        probe.train()
        order = list(range(len(train)))
        for _ in range(cfg.epochs):
            perm = torch.randperm(len(order), generator=gen).tolist()
            for pi in perm:
                ex = train[order[pi]]
                feats = ex.features.float().to(device)
                ii, jj, y = _sample_pairs(ex, pairs_per_protein, gen)
                if ii.numel() == 0:
                    continue
                opt.zero_grad(set_to_none=True)
                logits = probe.pair_logits(feats, ii, jj)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                opt.step()
        val_metrics = _contact_eval(probe, val, device)
        all_val[cfg.tag()] = val_metrics["p_at_l5_mean"]
        if best is None or val_metrics["p_at_l5_mean"] > best[0]:
            best = (val_metrics["p_at_l5_mean"], cfg, probe)
    assert best is not None, "empty probe grid"
    val_metric, cfg, probe = best
    return ProbeResult(
        task="contact",
        best_config=cfg,
        val_metric=val_metric,
        test_metrics=_contact_eval(probe, test, device),
        all_val_metrics=all_val,
    )


# --------------------------------------------------------------------------------------
# Task 3: mean-pooled regression (fluorescence, stability)
# --------------------------------------------------------------------------------------


@dataclass
class PooledTaskData:
    """Mean-pooled features and scalar targets for a regression task."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    @classmethod
    def from_splits(cls, splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> "PooledTaskData":
        need = ("train", "val", "test")
        missing = [k for k in need if k not in splits]
        if missing:
            raise KeyError(f"missing splits: {missing}")
        return cls(
            train_x=splits["train"][0].float(),
            train_y=splits["train"][1].float().reshape(-1),
            val_x=splits["val"][0].float(),
            val_y=splits["val"][1].float().reshape(-1),
            test_x=splits["test"][0].float(),
            test_y=splits["test"][1].float().reshape(-1),
        )


def run_regression_probe(
    data: PooledTaskData,
    task: str = "fluorescence",
    grid: Sequence[ProbeConfig] = DEFAULT_GRID,
    device: Optional[torch.device] = None,
    seed: int = 0,
    standardise: bool = True,
) -> ProbeResult:
    """Sweep the fixed grid for a mean-pooled scalar regression; report Spearman rho.

    Features are standardised using *train* statistics only (the val/test
    statistics never touch the fit), which matters because pooled embedding scales
    differ wildly between a collapsed and a healthy encoder and we do not want the
    probe optimiser's step size to be the thing that distinguishes conditions.

    Args:
        data: pooled features and targets.
        task: name recorded in the result.
        grid: shared hyperparameter grid.
        device: torch device.
        seed: probe seed.
        standardise: z-score features with train mean/std.

    Returns:
        :class:`ProbeResult` with ``test_metrics = {"spearman": ..., "mse": ...}``.
    """
    device = device or torch.device("cpu")
    tx, vx, sx = data.train_x, data.val_x, data.test_x
    if standardise:
        mu = tx.mean(0, keepdim=True)
        sd = tx.std(0, keepdim=True).clamp_min(1e-6)
        tx, vx, sx = (tx - mu) / sd, (vx - mu) / sd, (sx - mu) / sd
    y_mu, y_sd = data.train_y.mean(), data.train_y.std().clamp_min(1e-6)
    ty = (data.train_y - y_mu) / y_sd

    in_dim = tx.shape[-1]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda p, t: F.mse_loss(
        p.squeeze(-1), t
    )
    best: Optional[Tuple[float, ProbeConfig, nn.Module]] = None
    all_val: Dict[str, float] = {}
    for i, cfg in enumerate(grid):
        torch.manual_seed(seed + i)
        head = _fit_head(
            _build_head(in_dim, 1, cfg), tx, ty, cfg, loss_fn, device, seed=seed + i
        )
        with torch.no_grad():
            val_pred = head(vx.to(device)).squeeze(-1)
        rho = spearman_rho(val_pred, data.val_y)
        all_val[cfg.tag()] = rho
        if best is None or rho > best[0]:
            best = (rho, cfg, head)
    assert best is not None, "empty probe grid"
    val_rho, cfg, head = best
    with torch.no_grad():
        test_pred = head(sx.to(device)).squeeze(-1).cpu()
    denorm = test_pred * y_sd + y_mu
    return ProbeResult(
        task=task,
        best_config=cfg,
        val_metric=val_rho,
        test_metrics={
            "spearman": spearman_rho(test_pred, data.test_y),
            "mse": float(F.mse_loss(denorm, data.test_y)),
        },
        all_val_metrics=all_val,
    )
