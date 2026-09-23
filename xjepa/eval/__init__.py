"""Evaluation suite: collapse diagnostics, frozen-feature probes, zero-shot retrieval.

Nothing here is imported by the training step itself except
:class:`~xjepa.eval.collapse.CollapseMonitor`, which the trainer calls on its
logging cadence only.
"""

from __future__ import annotations

from .collapse import (
    RANKME_EPS,
    CollapseMonitor,
    collapse_metrics,
    dim_collapse_fraction,
    dim_std,
    offdiag_cov_mass,
    rankme,
)
from .probes import (
    DEFAULT_GRID,
    LINEAR_GRID,
    MLP_GRID,
    BilinearContactProbe,
    ContactExample,
    FeatureCache,
    LinearProbe,
    MlpProbe,
    PooledTaskData,
    ProbeConfig,
    ProbeResult,
    ResidueTaskData,
    extract_features,
    mean_pool,
    precision_at_l_over_k,
    run_contact_probe,
    run_regression_probe,
    run_residue_classification_probe,
    spearman_rho,
)
from .retrieval import (
    RetrievalResult,
    VariantEffectResult,
    cosine_similarity_matrix,
    fold_retrieval,
    l2_normalise,
    score_dms_assay,
    superfamily_disjoint_split,
    variant_effect_scores,
)

__all__ = [
    # collapse
    "RANKME_EPS",
    "rankme",
    "dim_std",
    "dim_collapse_fraction",
    "offdiag_cov_mass",
    "collapse_metrics",
    "CollapseMonitor",
    # probes
    "ProbeConfig",
    "ProbeResult",
    "LINEAR_GRID",
    "MLP_GRID",
    "DEFAULT_GRID",
    "LinearProbe",
    "MlpProbe",
    "BilinearContactProbe",
    "ResidueTaskData",
    "ContactExample",
    "PooledTaskData",
    "FeatureCache",
    "extract_features",
    "mean_pool",
    "spearman_rho",
    "precision_at_l_over_k",
    "run_residue_classification_probe",
    "run_contact_probe",
    "run_regression_probe",
    # retrieval
    "RetrievalResult",
    "VariantEffectResult",
    "fold_retrieval",
    "superfamily_disjoint_split",
    "cosine_similarity_matrix",
    "l2_normalise",
    "variant_effect_scores",
    "score_dms_assay",
]
