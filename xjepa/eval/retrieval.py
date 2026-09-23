"""Zero-shot evaluations: fold retrieval and a ProteinGym-style variant scorer.

Neither of these trains anything -- they read the frozen mean-pooled (or
per-residue) embeddings straight out of the feature cache, which makes them the
cheapest rows in the eval table and the only ones that are objective-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .probes import mean_pool, spearman_rho

__all__ = [
    "mean_pool",
    "l2_normalise",
    "cosine_similarity_matrix",
    "superfamily_disjoint_split",
    "RetrievalResult",
    "fold_retrieval",
    "VariantEffectResult",
    "variant_effect_scores",
    "score_dms_assay",
]


def l2_normalise(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise L2 normalisation of a ``[..., D]`` tensor."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_similarity_matrix(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """``[Q, D] x [G, D] -> [Q, G]`` cosine similarities (fp32 internally)."""
    q = l2_normalise(queries.float())
    g = l2_normalise(gallery.float())
    return q @ g.transpose(0, 1)


# --------------------------------------------------------------------------------------
# Superfamily-disjoint splitting (RESEARCH_PLAN.md 1.6, L2)
# --------------------------------------------------------------------------------------


def superfamily_disjoint_split(
    fold_labels: Sequence[int] | torch.Tensor,
    superfamily_labels: Sequence[int] | torch.Tensor,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a SCOPe-style set so no superfamily appears on both sides.

    A random query/gallery split leaks: near-identical domains from the same
    superfamily end up on both sides and retrieval becomes a near-duplicate
    lookup. Here every *superfamily* is assigned wholesale to either query or
    gallery, and only folds that still have superfamilies on both sides
    contribute queries. Folds with a single superfamily can never be scored
    fairly, so their domains are kept as gallery distractors instead of being
    dropped (dropping them would make the task easier than it is).

    Args:
        fold_labels: ``[N]`` fold id per domain.
        superfamily_labels: ``[N]`` superfamily id per domain (finer than fold).
        seed: controls which superfamily of each fold becomes the query side.

    Returns:
        ``(query_idx, gallery_idx)``, int64 index tensors into the domain array.
        Every query's fold is guaranteed to be represented in the gallery by at
        least one *different* superfamily, so top-1 accuracy is well defined.
    """
    folds = torch.as_tensor(fold_labels).reshape(-1).long()
    sfams = torch.as_tensor(superfamily_labels).reshape(-1).long()
    if folds.numel() != sfams.numel():
        raise ValueError("fold_labels and superfamily_labels must be the same length")
    gen = torch.Generator().manual_seed(seed)

    query_sfams: set[int] = set()
    for fold in folds.unique().tolist():
        in_fold = folds == fold
        fold_sfams = sfams[in_fold].unique()
        if fold_sfams.numel() < 2:
            continue  # cannot be queried without leaking; stays gallery-only
        pick = int(torch.randint(fold_sfams.numel(), (1,), generator=gen))
        query_sfams.add(int(fold_sfams[pick]))

    is_query = torch.tensor([int(s) in query_sfams for s in sfams.tolist()], dtype=torch.bool)
    query_idx = is_query.nonzero(as_tuple=False).squeeze(-1)
    gallery_idx = (~is_query).nonzero(as_tuple=False).squeeze(-1)
    return query_idx.long(), gallery_idx.long()


@dataclass
class RetrievalResult:
    """Fold-retrieval scores."""

    top1_accuracy: float
    mean_average_precision: float
    n_queries: int
    n_gallery: int
    per_query_top1: torch.Tensor = field(repr=False, default_factory=lambda: torch.empty(0))

    def to_dict(self) -> Dict[str, float]:
        return {
            "top1_accuracy": self.top1_accuracy,
            "map": self.mean_average_precision,
            "n_queries": float(self.n_queries),
            "n_gallery": float(self.n_gallery),
        }


def fold_retrieval(
    embeddings: torch.Tensor,
    fold_labels: Sequence[int] | torch.Tensor,
    superfamily_labels: Optional[Sequence[int] | torch.Tensor] = None,
    query_idx: Optional[torch.Tensor] = None,
    gallery_idx: Optional[torch.Tensor] = None,
    seed: int = 0,
    chunk_size: int = 512,
) -> RetrievalResult:
    """Zero-shot fold retrieval by cosine similarity over mean-pooled embeddings.

    Args:
        embeddings: ``[N, D]`` pooled embeddings (use
            :func:`xjepa.eval.probes.mean_pool` on encoder output). fp16 is
            promoted to fp32 before normalisation.
        fold_labels: ``[N]`` fold id per domain; a retrieval is correct when the
            retrieved domain shares the query's fold.
        superfamily_labels: ``[N]`` superfamily ids. When given (and no explicit
            indices are passed) the split is made superfamily-disjoint via
            :func:`superfamily_disjoint_split`.
        query_idx: explicit query indices (overrides the automatic split).
        gallery_idx: explicit gallery indices.
        seed: split seed.
        chunk_size: queries scored per similarity chunk, to bound peak memory.

    Returns:
        :class:`RetrievalResult` with top-1 accuracy and MAP (mean over queries
        of average precision over the full gallery ranking).

    Raises:
        ValueError: if the resulting split leaves no queries or no gallery.
    """
    emb = embeddings.reshape(embeddings.shape[0], -1).float()
    folds = torch.as_tensor(fold_labels).reshape(-1).long()
    if emb.shape[0] != folds.numel():
        raise ValueError("embeddings and fold_labels disagree on N")

    if query_idx is None or gallery_idx is None:
        if superfamily_labels is None:
            raise ValueError(
                "provide superfamily_labels for a disjoint split, or explicit "
                "query_idx/gallery_idx"
            )
        query_idx, gallery_idx = superfamily_disjoint_split(folds, superfamily_labels, seed=seed)
    if query_idx.numel() == 0 or gallery_idx.numel() == 0:
        raise ValueError("empty query or gallery split")

    q_emb = l2_normalise(emb.index_select(0, query_idx))
    g_emb = l2_normalise(emb.index_select(0, gallery_idx))
    q_fold = folds.index_select(0, query_idx)
    g_fold = folds.index_select(0, gallery_idx)

    n_q = int(q_emb.shape[0])
    n_g = int(g_emb.shape[0])
    top1 = torch.zeros(n_q)
    aps = torch.zeros(n_q)
    ranks = torch.arange(1, n_g + 1, dtype=torch.float32)

    for start in range(0, n_q, chunk_size):
        stop = min(start + chunk_size, n_q)
        sims = q_emb[start:stop] @ g_emb.transpose(0, 1)  # [c, G]
        order = sims.argsort(dim=1, descending=True)
        rel = (g_fold[order] == q_fold[start:stop].unsqueeze(1)).float()  # [c, G]
        top1[start:stop] = rel[:, 0]
        cum_hits = rel.cumsum(dim=1)
        precision_at_k = cum_hits / ranks
        n_rel = rel.sum(dim=1).clamp_min(1.0)
        aps[start:stop] = (precision_at_k * rel).sum(dim=1) / n_rel

    return RetrievalResult(
        top1_accuracy=float(top1.mean()),
        mean_average_precision=float(aps.mean()),
        n_queries=n_q,
        n_gallery=n_g,
        per_query_top1=top1,
    )


# --------------------------------------------------------------------------------------
# ProteinGym-style variant scoring
# --------------------------------------------------------------------------------------


@dataclass
class VariantEffectResult:
    """Spearman rho of embedding-distance variant scores against a DMS assay."""

    assay: str
    spearman: float
    n_variants: int

    def to_dict(self) -> Dict[str, float | str]:
        return {"assay": self.assay, "spearman": self.spearman, "n_variants": float(self.n_variants)}


def variant_effect_scores(
    wt_residue_embeddings: torch.Tensor,
    mut_residue_embeddings: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Score point mutants by cosine similarity at the mutated position.

    For variant ``v`` mutating position ``p_v``::

        score_v = cos( h_wt[p_v], h_mut_v[p_v] )

    i.e. the *negated* cosine distance, so that a larger score means "the
    encoder's local representation barely moved", which we take as the proxy for
    "the mutation is tolerated". Correlate directly (not inverted) against a
    fitness-style DMS label.

    .. warning::

       **This is exploratory and is expected to be weak: |rho| < 0.2.**
       It is pre-registered as such in RESEARCH_PLAN.md 5.2 and must be reported
       that way. A JEPA-only model exposes no token likelihoods, so the standard
       masked-marginal / pseudo-log-likelihood scoring used by ESM-style
       evaluations is simply unavailable here; cosine displacement in embedding
       space is not a fitness score and there is no reason to expect it to behave
       like one. Do not let any hypothesis rest on this number. For condition C1
       (MLM), score the same assays by pseudo-likelihood as well and label it
       clearly as a *different scoring function*, not a like-for-like comparison.

    Args:
        wt_residue_embeddings: ``[L, D]`` per-residue embeddings of the wild type.
        mut_residue_embeddings: ``[V, L, D]`` per-residue embeddings of each
            mutant sequence (same length L -- substitutions only).
        positions: ``[V]`` 0-based mutated position per variant.

    Returns:
        ``[V]`` float32 scores.

    Raises:
        ValueError: on shape mismatch or an out-of-range position.
    """
    wt = wt_residue_embeddings.float()
    mut = mut_residue_embeddings.float()
    if wt.ndim != 2 or mut.ndim != 3:
        raise ValueError("expected wt [L, D] and mutants [V, L, D]")
    if mut.shape[1:] != wt.shape:
        raise ValueError(f"mutant shape {tuple(mut.shape)} incompatible with wt {tuple(wt.shape)}")
    pos = torch.as_tensor(positions, dtype=torch.long).reshape(-1)
    if pos.numel() != mut.shape[0]:
        raise ValueError("positions must have one entry per mutant")
    if int(pos.min()) < 0 or int(pos.max()) >= wt.shape[0]:
        raise ValueError("mutated position out of range for the wild-type length")

    wt_at = wt.index_select(0, pos)  # [V, D]
    mut_at = mut[torch.arange(mut.shape[0]), pos]  # [V, D]
    return (l2_normalise(wt_at) * l2_normalise(mut_at)).sum(-1)


def score_dms_assay(
    wt_residue_embeddings: torch.Tensor,
    mut_residue_embeddings: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    dms_scores: torch.Tensor | Sequence[float],
    assay: str = "dms",
) -> VariantEffectResult:
    """:func:`variant_effect_scores` + Spearman rho against the assay labels.

    See the warning on :func:`variant_effect_scores`: expect |rho| < 0.2 and
    report it as exploratory.

    Args:
        wt_residue_embeddings: ``[L, D]``.
        mut_residue_embeddings: ``[V, L, D]``.
        positions: ``[V]`` mutated positions.
        dms_scores: ``[V]`` experimental fitness values (higher = fitter).
        assay: assay name for the record.

    Returns:
        :class:`VariantEffectResult`.
    """
    scores = variant_effect_scores(wt_residue_embeddings, mut_residue_embeddings, positions)
    labels = torch.as_tensor(dms_scores, dtype=torch.float32).reshape(-1)
    if labels.numel() != scores.numel():
        raise ValueError("dms_scores must have one entry per mutant")
    return VariantEffectResult(
        assay=assay, spearman=spearman_rho(scores, labels), n_variants=int(scores.numel())
    )
