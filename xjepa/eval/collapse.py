"""Collapse diagnostics for joint-embedding pretraining.

Three complementary statistics, all computed on a *fixed* held-out probe batch so
that values are comparable across steps, seeds and conditions:

* :func:`rankme`            -- soft/effective rank (Garrido et al., 2023, arXiv:2210.02885).
* :func:`dim_std`           -- VICReg variance criterion (Bardes et al., 2022, arXiv:2105.04906).
* :func:`offdiag_cov_mass`  -- fraction of covariance Frobenius mass sitting off the
  diagonal. RankMe smooths over *dimensional* collapse (a handful of strongly
  correlated directions can still produce a respectable singular-value entropy);
  this does not.

Cost discipline
---------------
These run inside the training loop at a fixed cadence (every ``log_every`` steps,
see RESEARCH_PLAN.md 5.1). :class:`CollapseMonitor` therefore holds *one*
pre-allocated probe batch on device, and one *pre-computed* index tensor used to
subsample residue embeddings down to a fixed ``n_vectors``. Nothing in the hot
path allocates as a function of the number of calls. The SVD is over an
``[n_vectors, d_model]`` matrix (8192 x 320 by default) which is sub-millisecond.

The functions themselves are *not* sync-free -- they return Python floats and are
meant to be called on the logging cadence only, never inside a training step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

__all__ = [
    "RANKME_EPS",
    "rankme",
    "dim_std",
    "dim_collapse_fraction",
    "offdiag_cov_mass",
    "collapse_metrics",
    "CollapseMonitor",
]

# Epsilon convention (Garrido et al., 2023, eq. 2).
#
# RankMe(Z) = exp(-sum_k p_k log p_k)  with  p_k = sigma_k / ||sigma||_1 + eps
#
# The epsilon is added to the *normalised* singular value p_k, NOT to the
# denominator and NOT inside the log. Two consequences worth stating because
# they are easy to get wrong:
#   1. The p_k no longer sum to exactly 1 (they sum to 1 + d*eps). The original
#      paper accepts this; we keep it so numbers are directly comparable to
#      published RankMe values.
#   2. eps = 1e-7 is the paper's value. For a rank-1 matrix it contributes
#      (d-1) * eps * log(1/eps) to the entropy, i.e. ~2.6e-5 at d = 2048, so
#      RankMe of a collapsed representation reads 1.000 rather than exactly 1.
RANKME_EPS: float = 1e-7


def _as_2d_float(x: torch.Tensor) -> torch.Tensor:
    """Flatten leading dims and promote to fp32 for numerically stable linalg."""
    if x.ndim < 2:
        raise ValueError(f"expected at least a 2-D tensor, got shape {tuple(x.shape)}")
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    elif not x.is_floating_point():
        x = x.float()
    return x


def rankme(x: torch.Tensor, eps: float = RANKME_EPS) -> float:
    """RankMe: the exponentiated Shannon entropy of the normalised spectrum.

    Args:
        x: ``[N, D]`` (or ``[..., D]``, leading dims are flattened) matrix of
            embeddings. fp16/bf16 inputs are promoted to fp32.
        eps: smoothing added to each normalised singular value; see
            :data:`RANKME_EPS` for the convention.

    Returns:
        A float in ``[1, min(N, D)]``. 1.0 means a fully collapsed (rank-1)
        representation; a value near ``D`` means an isotropic one.

    Notes:
        Uses :func:`torch.linalg.svdvals`, which is the cheap path (no singular
        vectors computed). Cost is O(N D^2) -- keep N modest (8192 is plenty).
    """
    x = _as_2d_float(x)
    svals = torch.linalg.svdvals(x)
    svals = svals.clamp_min(0.0)
    total = svals.sum()
    if float(total) <= 0.0:
        # Degenerate: the all-zeros representation. Rank is 1 by convention.
        return 1.0
    p = svals / total + eps
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


def dim_std(x: torch.Tensor, eps: float = 1e-4) -> float:
    """VICReg variance criterion: mean per-dimension standard deviation.

    Bardes et al. (2022) regularise ``sqrt(Var(z_j) + eps)`` per dimension; we
    report its mean over dimensions, which is the quantity that hits the hinge.
    A healthy representation sits well above the hinge threshold (1.0 in VICReg,
    though that is scale-dependent and we do not normalise here); values decaying
    towards 0 are the signature of a fully collapsed encoder.

    Args:
        x: ``[N, D]`` embeddings, leading dims flattened.
        eps: variance floor inside the square root (VICReg's ``eps``).

    Returns:
        Mean over ``D`` of ``sqrt(var_j + eps)``, using the unbiased variance.
    """
    x = _as_2d_float(x)
    if x.shape[0] < 2:
        raise ValueError("dim_std needs at least 2 rows to estimate a variance")
    var = x.var(dim=0, unbiased=True)
    return float(torch.sqrt(var + eps).mean())


def dim_collapse_fraction(x: torch.Tensor, threshold: float = 1e-2) -> float:
    """Fraction of dimensions whose std falls below ``threshold``.

    Partial/dimensional collapse shows up here long before the mean std moves.

    Args:
        x: ``[N, D]`` embeddings.
        threshold: std below which a dimension counts as dead (VICReg uses 1e-2
            style thresholds when reporting; RESEARCH_PLAN.md 5.1 asks for 0.01).

    Returns:
        Fraction in ``[0, 1]``.
    """
    x = _as_2d_float(x)
    if x.shape[0] < 2:
        raise ValueError("dim_collapse_fraction needs at least 2 rows")
    std = x.std(dim=0, unbiased=True)
    return float((std < threshold).float().mean())


def offdiag_cov_mass(x: torch.Tensor) -> float:
    """Fraction of the covariance matrix's squared Frobenius mass that is off-diagonal.

    Let ``C = cov(x)`` (``[D, D]``, unbiased). We return

        ``(||C||_F^2 - ||diag(C)||_2^2) / ||C||_F^2``

    i.e. *squared* Frobenius mass, which is the additive energy decomposition and
    the convention used by the covariance term in VICReg (which penalises the sum
    of squared off-diagonal entries).

    Interpretation:
        * 0.0  -- perfectly decorrelated dimensions (identity-shaped covariance).
        * -> 1 -- every dimension carries the same direction, i.e. dimensional
          collapse. For ``D`` identical dimensions the value is ``(D-1)/D``.

    Args:
        x: ``[N, D]`` embeddings, leading dims flattened.

    Returns:
        Float in ``[0, 1)``. Returns 0.0 for a numerically zero covariance.
    """
    x = _as_2d_float(x)
    n, d = x.shape
    if n < 2:
        raise ValueError("offdiag_cov_mass needs at least 2 rows")
    if d < 2:
        return 0.0
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / (n - 1)
    total = (cov * cov).sum()
    if float(total) <= 0.0:
        return 0.0
    diag = torch.diagonal(cov)
    diag_mass = (diag * diag).sum()
    return float(((total - diag_mass) / total).clamp(0.0, 1.0))


def collapse_metrics(x: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    """All four diagnostics for one embedding matrix, as a flat dict of floats.

    Args:
        x: ``[N, D]`` (or ``[B, L, D]``) embeddings.
        prefix: key prefix, e.g. ``"enc/"`` or ``"pred/"``.

    Returns:
        ``{prefix+"rankme", prefix+"dim_std", prefix+"dead_dims",
        prefix+"offdiag_cov_mass"}``.
    """
    x = _as_2d_float(x)
    return {
        f"{prefix}rankme": rankme(x),
        f"{prefix}dim_std": dim_std(x),
        f"{prefix}dead_dims": dim_collapse_fraction(x),
        f"{prefix}offdiag_cov_mass": offdiag_cov_mass(x),
    }


@dataclass
class CollapseMonitor:
    """Holds a fixed probe batch on device and scores an encoder against it.

    The batch is fixed for the lifetime of a run (and, if you build it from the
    same held-out slice with the same seed, across runs) so that the RankMe trace
    is a property of the model and not of the data it happened to see.

    Args:
        tokens: ``[B, L]`` int64 token ids, already on the target device.
        pad_mask: ``[B, L]`` bool, True = real residue (matches the ``Batch``
            contract in docs/CONTRACTS.md).
        n_vectors: number of residue embeddings kept for the statistics. The
            selection indices are drawn once, at construction, from the *valid*
            (non-pad) positions, and reused for every call.
        seed: seed for that one-off selection.

    Attributes:
        index: ``[n_vectors]`` int64 flat indices into the ``[B*L, D]`` view of
            the encoder output. Pre-allocated; never regrown.
    """

    tokens: torch.Tensor
    pad_mask: torch.Tensor
    n_vectors: int = 8192
    seed: int = 0
    index: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2 or self.pad_mask.shape != self.tokens.shape:
            raise ValueError("tokens and pad_mask must both be [B, L] and matching")
        device = self.tokens.device
        flat_valid = self.pad_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n_valid = int(flat_valid.numel())
        if n_valid == 0:
            raise ValueError("probe batch contains no unpadded residues")
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        k = min(self.n_vectors, n_valid)
        perm = torch.randperm(n_valid, generator=gen)[:k]
        self.index = flat_valid[perm.to(device)].contiguous()
        self.n_vectors = k

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    def gather(self, features: torch.Tensor) -> torch.Tensor:
        """Select the fixed residue subset out of a ``[B, L, D]`` feature tensor."""
        if features.ndim != 3:
            raise ValueError(f"expected [B, L, D] features, got {tuple(features.shape)}")
        flat = features.reshape(-1, features.shape[-1])
        return flat.index_select(0, self.index)

    @torch.no_grad()
    def measure(
        self,
        feature_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        prefix: str = "",
    ) -> Dict[str, float]:
        """Run ``feature_fn(tokens, pad_mask) -> [B, L, D]`` and score its output.

        Args:
            feature_fn: usually ``encoder`` itself, or a lambda that additionally
                pushes the encoder output through the predictor. RESEARCH_PLAN.md
                1.1c asks for the *encoder* trace as primary, predictor separately.
            prefix: metric-key prefix.

        Returns:
            Dict of Python floats (safe: this is off the training hot path).
        """
        feats = feature_fn(self.tokens, self.pad_mask)
        return collapse_metrics(self.gather(feats), prefix=prefix)

    @torch.no_grad()
    def measure_module(self, module: torch.nn.Module, prefix: str = "enc/") -> Dict[str, float]:
        """Convenience wrapper: put ``module`` in eval mode, score, restore mode."""
        was_training = module.training
        module.eval()
        try:
            return self.measure(lambda t, m: module(t, m), prefix=prefix)
        finally:
            module.train(was_training)

    @staticmethod
    def reference_rank(bank: torch.Tensor, n_vectors: int = 8192, seed: int = 0) -> float:
        """RankMe of a raw target bank -- the ceiling line for H1b.

        Args:
            bank: ``[N, D]`` target embeddings (e.g. the PCA-128 ESM-IF1 cache).
            n_vectors: random subsample size, for cost parity with the live traces.
            seed: subsample seed.

        Returns:
            RankMe of the subsample.
        """
        flat = _as_2d_float(bank)
        n = flat.shape[0]
        if n > n_vectors:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            idx = torch.randperm(n, generator=gen)[:n_vectors].to(flat.device)
            flat = flat.index_select(0, idx)
        return rankme(flat)
