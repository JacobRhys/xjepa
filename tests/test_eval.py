"""CPU tests for the eval suite and the performance harness.

Everything here runs on CPU with tiny shapes (docs/CONTRACTS.md, hard rule 7) and
should finish in well under a minute on a laptop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from xjepa.eval.collapse import (
    CollapseMonitor,
    collapse_metrics,
    dim_collapse_fraction,
    dim_std,
    offdiag_cov_mass,
    rankme,
)
from xjepa.eval.probes import (
    LINEAR_GRID,
    BilinearContactProbe,
    ContactExample,
    FeatureCache,
    PooledTaskData,
    ProbeConfig,
    ResidueTaskData,
    extract_features,
    mean_pool,
    precision_at_l_over_k,
    run_contact_probe,
    run_regression_probe,
    run_residue_classification_probe,
    spearman_rho,
)
from xjepa.eval.retrieval import (
    fold_retrieval,
    score_dms_assay,
    superfamily_disjoint_split,
    variant_effect_scores,
)
from xjepa.perf.bench import (
    REFERENCE_PARAMS,
    BenchConfig,
    SyntheticCorpus,
    detect_sdpa_backend,
    peak_bf16_tflops,
    run_benchmark,
    transformer_flops_per_step,
)
from xjepa.perf.profile_report import run_profile

CPU = torch.device("cpu")


# ======================================================================================
# collapse.py
# ======================================================================================


def test_rankme_is_one_for_rank_one_matrix() -> None:
    """A collapsed (rank-1) representation must read RankMe == 1."""
    torch.manual_seed(0)
    direction = torch.randn(32)
    scales = torch.randn(512, 1)
    x = scales * direction  # every row is a multiple of one vector
    assert rankme(x) == pytest.approx(1.0, abs=1e-3)


def test_rankme_is_one_for_constant_rows() -> None:
    """The literal degenerate case: every embedding identical."""
    x = torch.ones(256, 16) * 3.0
    assert rankme(x) == pytest.approx(1.0, abs=1e-3)


def test_rankme_approaches_d_for_isotropic_gaussian() -> None:
    """An isotropic Gaussian has near-maximal soft rank."""
    torch.manual_seed(0)
    d = 32
    x = torch.randn(8192, d)
    r = rankme(x)
    assert 0.9 * d <= r <= d + 1e-6, f"rankme={r} for isotropic d={d}"


def test_rankme_monotone_in_true_rank() -> None:
    """More independent directions -> strictly larger RankMe."""
    torch.manual_seed(0)
    d = 32
    ranks = [1, 4, 16, 32]
    vals = []
    for k in ranks:
        basis = torch.linalg.qr(torch.randn(d, d))[0][:, :k]
        x = torch.randn(4096, k) @ basis.T
        vals.append(rankme(x))
    assert all(a < b for a, b in zip(vals, vals[1:])), vals
    assert vals[0] == pytest.approx(1.0, abs=1e-3)


def test_rankme_handles_zero_matrix() -> None:
    assert rankme(torch.zeros(16, 8)) == 1.0


def test_rankme_flattens_leading_dims_and_promotes_dtype() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 64, 16, dtype=torch.float16)
    r = rankme(x)
    assert 1.0 < r <= 16.0


def test_dim_std_matches_hand_computed_value() -> None:
    """dim_std is the mean of sqrt(var + eps) per dimension."""
    x = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    # dim 0: unbiased var of [0,2,4,6] = 6.6667 -> std 2.582; dim 1: 0
    expected = 0.5 * (math.sqrt(6.6666667 + 1e-4) + math.sqrt(1e-4))
    assert dim_std(x) == pytest.approx(expected, rel=1e-4)


def test_dim_std_collapses_to_zero() -> None:
    x = torch.ones(128, 8)
    assert dim_std(x) < 0.02


def test_dim_std_scales_with_spread() -> None:
    torch.manual_seed(0)
    base = torch.randn(1024, 8)
    assert dim_std(base * 5.0) > 4.0 * dim_std(base)


def test_dim_collapse_fraction_counts_dead_dimensions() -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, 10)
    x[:, :4] = 0.0  # four dead dimensions
    assert dim_collapse_fraction(x, threshold=1e-2) == pytest.approx(0.4)


def test_offdiag_cov_mass_zero_for_independent_dims() -> None:
    torch.manual_seed(0)
    x = torch.randn(20000, 8)
    assert offdiag_cov_mass(x) < 0.05


def test_offdiag_cov_mass_near_one_for_duplicated_dims() -> None:
    """D identical columns => off-diagonal mass is exactly (D-1)/D."""
    torch.manual_seed(0)
    col = torch.randn(512, 1)
    d = 8
    x = col.repeat(1, d)
    assert offdiag_cov_mass(x) == pytest.approx((d - 1) / d, rel=1e-4)


def test_offdiag_cov_mass_catches_what_rankme_smooths() -> None:
    """Two strongly correlated blocks: RankMe stays high-ish, off-diag mass spikes."""
    torch.manual_seed(0)
    z = torch.randn(4096, 8)
    correlated = torch.cat([z, z + 0.01 * torch.randn(4096, 8)], dim=1)  # 16 dims, rank ~8
    assert rankme(correlated) > 5.0
    assert offdiag_cov_mass(correlated) > 0.4
    assert offdiag_cov_mass(torch.randn(4096, 16)) < 0.05


def test_offdiag_cov_mass_single_dimension_is_zero() -> None:
    assert offdiag_cov_mass(torch.randn(64, 1)) == 0.0


def test_collapse_metrics_keys_and_prefix() -> None:
    m = collapse_metrics(torch.randn(256, 8), prefix="enc/")
    assert set(m) == {"enc/rankme", "enc/dim_std", "enc/dead_dims", "enc/offdiag_cov_mass"}
    assert all(isinstance(v, float) for v in m.values())


class _ToyEncoder(torch.nn.Module):
    """Tiny stand-in for the real encoder: embedding + one linear."""

    def __init__(self, vocab: int = 33, d: int = 16, collapse: bool = False) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.lin = torch.nn.Linear(d, d)
        self.collapse = collapse

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.lin(self.emb(tokens))
        if self.collapse:
            h = torch.ones_like(h)
        return h


def test_collapse_monitor_is_fixed_and_bounded() -> None:
    torch.manual_seed(0)
    tokens = torch.randint(0, 33, (4, 32))
    pad_mask = torch.ones(4, 32, dtype=torch.bool)
    pad_mask[:, 24:] = False  # padding must be excluded from the probe set
    mon = CollapseMonitor(tokens=tokens, pad_mask=pad_mask, n_vectors=50, seed=0)

    assert mon.n_vectors == 50
    # index must only ever point at unpadded positions
    flat_valid = pad_mask.reshape(-1)
    assert bool(flat_valid[mon.index].all())

    enc = _ToyEncoder()
    before = mon.index.clone()
    m1 = mon.measure_module(enc)
    m2 = mon.measure_module(enc)
    assert torch.equal(before, mon.index), "probe selection must not change between calls"
    assert m1 == m2, "the same encoder on the same fixed batch must score identically"
    assert set(m1) == {"enc/rankme", "enc/dim_std", "enc/dead_dims", "enc/offdiag_cov_mass"}
    assert enc.training, "measure_module must restore the original train/eval mode"


def test_collapse_monitor_detects_a_collapsed_encoder() -> None:
    torch.manual_seed(0)
    tokens = torch.randint(0, 33, (4, 32))
    pad_mask = torch.ones(4, 32, dtype=torch.bool)
    mon = CollapseMonitor(tokens=tokens, pad_mask=pad_mask, n_vectors=100, seed=0)
    healthy = mon.measure_module(_ToyEncoder(collapse=False))
    dead = mon.measure_module(_ToyEncoder(collapse=True))
    assert dead["enc/rankme"] < healthy["enc/rankme"]
    assert dead["enc/rankme"] == pytest.approx(1.0, abs=1e-3)
    assert dead["enc/dead_dims"] == 1.0


def test_collapse_monitor_rejects_all_padding() -> None:
    with pytest.raises(ValueError):
        CollapseMonitor(
            tokens=torch.zeros(2, 4, dtype=torch.long),
            pad_mask=torch.zeros(2, 4, dtype=torch.bool),
        )


def test_reference_rank_of_target_bank() -> None:
    """The ceiling line for H1b: RankMe of the (subsampled) target bank."""
    torch.manual_seed(0)
    bank = torch.randn(20000, 32)
    r = CollapseMonitor.reference_rank(bank, n_vectors=4096, seed=0)
    assert 0.85 * 32 <= r <= 32 + 1e-6


# ======================================================================================
# probes.py
# ======================================================================================


def test_spearman_rho_perfect_and_inverted() -> None:
    x = torch.arange(20).float()
    assert spearman_rho(x, x) == pytest.approx(1.0)
    assert spearman_rho(x, -x) == pytest.approx(-1.0)
    assert spearman_rho(x, torch.ones(20)) == 0.0


def test_spearman_rho_is_rank_based() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([1.0, 10.0, 100.0, 1000.0])  # monotone but very non-linear
    assert spearman_rho(x, y) == pytest.approx(1.0)


def _separable_residue_task(
    n_proteins: int = 24, length: int = 16, d: int = 12, n_classes: int = 3, noise: float = 0.6
):
    """Build a linearly separable per-residue classification task."""
    g = torch.Generator().manual_seed(0)
    centroids = torch.randn(n_classes, d, generator=g) * 3.0

    def make(n: int):
        labels = torch.randint(0, n_classes, (n, length), generator=g)
        feats = centroids[labels] + noise * torch.randn(n, length, d, generator=g)
        return feats, labels

    return {"train": make(n_proteins), "val": make(8), "test": make(8)}, n_classes


def test_residue_classification_probe_beats_chance() -> None:
    splits, n_classes = _separable_residue_task()
    data = ResidueTaskData.from_padded(splits, num_classes=n_classes)
    grid = (ProbeConfig(lr=3e-3, weight_decay=1e-4, epochs=30, batch_size=64, hidden=None),)
    res = run_residue_classification_probe(data, task="ss3", grid=grid, device=CPU)
    assert res.task == "ss3"
    assert res.test_metrics["accuracy"] > 0.8, res.test_metrics
    assert res.val_metric > 1.0 / n_classes


def test_residue_probe_mlp_variant_also_learns() -> None:
    splits, n_classes = _separable_residue_task()
    data = ResidueTaskData.from_padded(splits, num_classes=n_classes)
    grid = (ProbeConfig(lr=3e-3, weight_decay=1e-4, epochs=30, batch_size=64, hidden=256),)
    res = run_residue_classification_probe(data, task="ss8", grid=grid, device=CPU)
    assert res.best_config.hidden == 256
    assert res.test_metrics["accuracy"] > 0.8


def test_residue_task_drops_ignore_index() -> None:
    feats = torch.randn(2, 5, 4)
    labels = torch.tensor([[0, 1, -100, -100, 2], [1, -100, 0, 2, 2]])
    splits = {k: (feats, labels) for k in ("train", "val", "test")}
    data = ResidueTaskData.from_padded(splits, num_classes=3)
    assert data.train_x.shape[0] == 7 == data.train_y.numel()
    assert int(data.train_y.min()) >= 0


def test_residue_task_requires_all_splits() -> None:
    with pytest.raises(KeyError):
        ResidueTaskData.from_padded({"train": (torch.randn(1, 2, 3), torch.zeros(1, 2).long())}, 2)


def test_probe_grid_is_shared_not_per_condition() -> None:
    """Structural check on the experimental control: one grid object, reused."""
    from xjepa.eval import probes as probes_mod

    assert probes_mod.DEFAULT_GRID == probes_mod.LINEAR_GRID + probes_mod.MLP_GRID
    assert all(c.hidden is None for c in probes_mod.LINEAR_GRID)
    assert all(c.hidden == 256 for c in probes_mod.MLP_GRID)
    assert len({c.tag() for c in probes_mod.DEFAULT_GRID}) == len(probes_mod.DEFAULT_GRID)


def _contact_examples(n: int, length: int = 64, d: int = 8, seed: int = 0):
    """Proteins whose contacts are a deterministic function of the features."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        feats = torch.randn(length, d, generator=g)
        # contact iff the two residues' first coordinates agree in sign
        sign = torch.sign(feats[:, 0])
        contacts = (sign.unsqueeze(0) * sign.unsqueeze(1)) > 0
        out.append(ContactExample(features=feats, contacts=contacts))
    return out


def test_contact_probe_learns_a_synthetic_rule() -> None:
    train = _contact_examples(6, seed=0)
    val = _contact_examples(3, seed=1)
    test = _contact_examples(3, seed=2)
    grid = (ProbeConfig(lr=1e-2, weight_decay=0.0, epochs=12, hidden=None),)
    res = run_contact_probe(train, val, test, grid=grid, pairs_per_protein=2000, rank=8, device=CPU)
    assert res.task == "contact"
    assert set(res.test_metrics) == {"p_at_l5_medium", "p_at_l5_long", "p_at_l5_mean"}
    # chance is ~0.5 for this construction; a working probe is far above it
    assert res.test_metrics["p_at_l5_long"] > 0.8, res.test_metrics


def test_precision_at_l5_is_perfect_for_oracle_scores() -> None:
    ex = _contact_examples(1, length=48, seed=3)[0]
    oracle = ex.contacts.float() * 100.0
    for band in ("medium", "long"):
        assert precision_at_l_over_k(oracle, ex, band=band, k=5) == pytest.approx(1.0)


def test_precision_at_l5_returns_none_for_empty_band() -> None:
    ex = _contact_examples(1, length=10, seed=4)[0]  # too short for long range
    assert precision_at_l_over_k(torch.zeros(10, 10), ex, band="long", k=5) is None
    # ... and the aggregate reports NaN rather than pretending the probe scored 0
    from xjepa.eval.probes import _contact_eval

    probe = BilinearContactProbe(8, ProbeConfig(hidden=None), rank=4)
    agg = _contact_eval(probe, [ex], CPU)
    assert math.isnan(agg["p_at_l5_long"])


def test_contact_probe_map_is_symmetric() -> None:
    probe = BilinearContactProbe(8, ProbeConfig(hidden=None), rank=4)
    feats = torch.randn(12, 8)
    m = probe.full_map(feats)
    assert torch.allclose(m, m.transpose(0, 1), atol=1e-6)


def test_regression_probe_recovers_a_linear_signal() -> None:
    g = torch.Generator().manual_seed(0)
    d = 10
    w = torch.randn(d, generator=g)

    def make(n: int):
        x = torch.randn(n, d, generator=g)
        y = x @ w + 0.1 * torch.randn(n, generator=g)
        return x, y

    data = PooledTaskData.from_splits({"train": make(400), "val": make(100), "test": make(100)})
    grid = (ProbeConfig(lr=1e-2, weight_decay=1e-5, epochs=40, batch_size=64, hidden=None),)
    res = run_regression_probe(data, task="fluorescence", grid=grid, device=CPU)
    assert res.test_metrics["spearman"] > 0.9, res.test_metrics
    assert res.task == "fluorescence"


def test_regression_probe_reports_chance_on_noise() -> None:
    g = torch.Generator().manual_seed(1)

    def make(n: int):
        return torch.randn(n, 6, generator=g), torch.randn(n, generator=g)

    data = PooledTaskData.from_splits({"train": make(200), "val": make(80), "test": make(80)})
    grid = (ProbeConfig(lr=1e-2, weight_decay=1e-2, epochs=10, batch_size=64, hidden=None),)
    res = run_regression_probe(data, grid=grid, device=CPU)
    assert abs(res.test_metrics["spearman"]) < 0.4


def test_full_grid_sweep_selects_on_validation() -> None:
    splits, n_classes = _separable_residue_task(n_proteins=10, length=8)
    data = ResidueTaskData.from_padded(splits, num_classes=n_classes)
    fast = tuple(
        ProbeConfig(lr=c.lr, weight_decay=c.weight_decay, epochs=5, batch_size=64)
        for c in LINEAR_GRID
    )
    res = run_residue_classification_probe(data, grid=fast, device=CPU)
    assert len(res.all_val_metrics) == len(fast)
    assert res.val_metric == max(res.all_val_metrics.values())
    assert res.best_config.tag() in res.all_val_metrics


def test_extract_features_shapes_and_pooling() -> None:
    enc = _ToyEncoder(d=16)
    tokens = torch.randint(0, 33, (7, 12))
    pad_mask = torch.ones(7, 12, dtype=torch.bool)
    pad_mask[:, 9:] = False
    per_res = extract_features(enc, tokens, pad_mask, batch_size=3, pool="residue")
    pooled = extract_features(enc, tokens, pad_mask, batch_size=3, pool="mean")
    assert per_res.shape == (7, 12, 16) and per_res.dtype == torch.float16
    assert pooled.shape == (7, 16)
    ref = mean_pool(per_res.float(), pad_mask)
    assert torch.allclose(pooled.float(), ref, atol=1e-2)


def test_extract_features_rejects_bad_pool() -> None:
    with pytest.raises(ValueError):
        extract_features(_ToyEncoder(), torch.zeros(1, 2).long(), torch.ones(1, 2).bool(), pool="max")


def test_feature_cache_extracts_once(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    calls = {"n": 0}

    def extract() -> torch.Tensor:
        calls["n"] += 1
        return torch.arange(12).reshape(3, 4).float()

    a = cache.get_or_extract("ckpt_c3_seed0", "cb513", extract, pool="residue")
    b = cache.get_or_extract("ckpt_c3_seed0", "cb513", extract, pool="residue")
    assert calls["n"] == 1, "features must be extracted exactly once per checkpoint/dataset"
    assert torch.equal(a, b)
    cache.get_or_extract("ckpt_c1_seed0", "cb513", extract, pool="residue")
    assert calls["n"] == 2, "a different checkpoint must miss the cache"
    manifest = cache.write_manifest("manifest", {"shape": list(a.shape)})
    assert json.loads(manifest.read_text())["shape"] == [3, 4]


# ======================================================================================
# retrieval.py
# ======================================================================================


def _clustered_embeddings(n_folds: int = 6, per_sfam: int = 4, sfams: int = 3, d: int = 16):
    """Tight, well-separated per-fold clusters with nested superfamilies."""
    g = torch.Generator().manual_seed(0)
    centres = torch.linalg.qr(torch.randn(d, d, generator=g))[0][:n_folds]  # orthogonal folds
    embs, folds, sfam_ids = [], [], []
    sid = 0
    for f in range(n_folds):
        for _ in range(sfams):
            for _ in range(per_sfam):
                embs.append(centres[f] + 0.01 * torch.randn(d, generator=g))
                folds.append(f)
                sfam_ids.append(sid)
            sid += 1
    return torch.stack(embs), torch.tensor(folds), torch.tensor(sfam_ids)


def test_fold_retrieval_is_perfect_on_clustered_data() -> None:
    emb, folds, sfams = _clustered_embeddings()
    res = fold_retrieval(emb, folds, sfams, seed=0)
    assert res.top1_accuracy == pytest.approx(1.0)
    assert res.mean_average_precision > 0.99
    assert res.n_queries > 0 and res.n_gallery > 0


def test_retrieval_split_is_superfamily_disjoint() -> None:
    _, folds, sfams = _clustered_embeddings()
    q, gal = superfamily_disjoint_split(folds, sfams, seed=0)
    assert set(sfams[q].tolist()).isdisjoint(set(sfams[gal].tolist()))
    # every query's fold must still be reachable in the gallery
    assert set(folds[q].tolist()).issubset(set(folds[gal].tolist()))


def test_single_superfamily_folds_stay_gallery_only() -> None:
    folds = torch.tensor([0, 0, 0, 1, 1])
    sfams = torch.tensor([0, 1, 1, 2, 2])  # fold 1 has one superfamily
    q, gal = superfamily_disjoint_split(folds, sfams, seed=0)
    assert 1 not in folds[q].tolist()
    assert set(q.tolist()).isdisjoint(set(gal.tolist()))
    assert sorted(q.tolist() + gal.tolist()) == list(range(5))


def test_fold_retrieval_is_chance_on_random_embeddings() -> None:
    torch.manual_seed(0)
    _, folds, sfams = _clustered_embeddings()
    emb = torch.randn(folds.numel(), 16)
    res = fold_retrieval(emb, folds, sfams, seed=0)
    assert res.top1_accuracy < 0.6  # 6 folds -> chance ~0.17


def test_fold_retrieval_requires_labels_or_indices() -> None:
    with pytest.raises(ValueError):
        fold_retrieval(torch.randn(4, 3), [0, 0, 1, 1])


def test_fold_retrieval_chunking_matches_single_shot() -> None:
    emb, folds, sfams = _clustered_embeddings()
    a = fold_retrieval(emb, folds, sfams, seed=0, chunk_size=3)
    b = fold_retrieval(emb, folds, sfams, seed=0, chunk_size=10_000)
    assert a.top1_accuracy == pytest.approx(b.top1_accuracy)
    assert a.mean_average_precision == pytest.approx(b.mean_average_precision, abs=1e-6)


def test_variant_effect_scores_are_cosine_at_the_mutated_position() -> None:
    torch.manual_seed(0)
    L, D, V = 10, 8, 4
    wt = torch.randn(L, D)
    mut = wt.unsqueeze(0).repeat(V, 1, 1)
    positions = torch.tensor([0, 3, 5, 9])
    # identical embeddings -> cosine 1 everywhere
    assert torch.allclose(variant_effect_scores(wt, mut, positions), torch.ones(V), atol=1e-5)
    # flip the sign at each mutated position -> cosine -1
    for v, p in enumerate(positions.tolist()):
        mut[v, p] = -wt[p]
    assert torch.allclose(variant_effect_scores(wt, mut, positions), -torch.ones(V), atol=1e-5)


def test_variant_effect_rejects_bad_shapes() -> None:
    wt = torch.randn(6, 4)
    with pytest.raises(ValueError):
        variant_effect_scores(wt, torch.randn(2, 5, 4), [0, 1])
    with pytest.raises(ValueError):
        variant_effect_scores(wt, torch.randn(2, 6, 4), [0, 99])
    with pytest.raises(ValueError):
        variant_effect_scores(wt, torch.randn(2, 6, 4), [0])


def test_score_dms_assay_returns_spearman() -> None:
    torch.manual_seed(0)
    L, D, V = 12, 8, 20
    wt = torch.randn(L, D)
    mut = wt.unsqueeze(0).repeat(V, 1, 1)
    positions = torch.randint(0, L, (V,))
    # construct a monotone relationship: bigger perturbation -> lower fitness
    strength = torch.linspace(0.0, 2.0, V)
    for v in range(V):
        mut[v, positions[v]] = wt[positions[v]] + strength[v] * torch.randn(D)
    labels = -strength
    res = score_dms_assay(wt, mut, positions, labels, assay="synthetic")
    assert res.assay == "synthetic" and res.n_variants == V
    assert -1.0 <= res.spearman <= 1.0


def test_proteingym_docstring_states_the_expected_weakness() -> None:
    """The exploratory caveat is part of the deliverable, not decoration."""
    doc = variant_effect_scores.__doc__ or ""
    assert "exploratory" in doc.lower()
    assert "0.2" in doc
    assert "likelihood" in doc.lower()


# ======================================================================================
# perf/bench.py + profile_report.py
# ======================================================================================


def test_flops_accounting_includes_attention_term() -> None:
    n, tokens, layers, d, L = 7_408_960, 1024, 6, 320, 512
    f = transformer_flops_per_step(n, tokens, layers, d, L)
    param_only = 6.0 * n * tokens
    assert f > param_only
    attn = 12.0 * layers * d * L * tokens
    assert f == pytest.approx(param_only + attn)
    # the attention term is ~21% of the total at this config
    assert 0.15 < attn / f < 0.25


def test_reference_param_counts_are_the_measured_ones() -> None:
    """Guard against the 8M round-up silently returning: it inflates MFU ~8%."""
    assert REFERENCE_PARAMS["encoder"] == 7_408_960
    assert REFERENCE_PARAMS["c2_c3_jepa"] - REFERENCE_PARAMS["encoder"] == pytest.approx(
        691_008, abs=1
    )
    assert REFERENCE_PARAMS["encoder"] < 8_000_000


def test_peak_lookup() -> None:
    assert peak_bf16_tflops("NVIDIA GeForce RTX 4090") == pytest.approx(165.2)
    assert peak_bf16_tflops("Tesla T4") is None  # no bf16
    assert peak_bf16_tflops("Some Unknown Accelerator") is None


def test_sdpa_backend_detection_degrades_on_cpu() -> None:
    b = detect_sdpa_backend(CPU, 2, 4, 16, 8, torch.float32, has_attn_mask=True)
    assert "cpu" in b.lower()


def test_synthetic_corpus_is_device_resident_and_index_batched() -> None:
    corpus = SyntheticCorpus(16, 8, 33, 4, CPU, seed=0)
    tokens, targets, pad_mask, mask_sel = corpus.batch(4)
    assert tokens.shape == (4, 8) and tokens.dtype == torch.int64
    assert targets.shape == (4, 8, 4) and targets.dtype == torch.float16
    assert pad_mask.shape == (4, 8) and pad_mask.dtype == torch.bool
    assert mask_sel.dtype == torch.bool
    assert corpus.tokens.dtype == torch.uint8
    assert corpus.nbytes == 16 * 8 * 1 + 16 * 8 * 4 * 2


def test_benchmark_runs_end_to_end_on_cpu() -> None:
    cfg = BenchConfig(
        steps=3,
        warmup=2,
        batch_size=2,
        seq_len=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        corpus_seqs=8,
        target_dim=8,
        device="cpu",
        measure_transfers=True,  # must degrade, not crash
    )
    res = run_benchmark(cfg, verbose=False)

    assert res.steps_per_sec > 0 and res.tokens_per_sec > 0
    assert res.tokens_per_sec == pytest.approx(res.steps_per_sec * 32, rel=1e-6)
    assert res.step_ms_mean > 0 and res.step_ms_p90 >= res.step_ms_p50 * 0.5
    assert res.fwd_ms > 0 and res.bwd_ms > 0 and res.opt_ms > 0
    assert res.other_ms >= 0
    assert res.flops_per_step > 0 and res.achieved_tflops > 0
    # GPU-only quantities must be absent with a clear explanation, not faked
    assert res.mfu is None and res.peak_bf16_tflops is None
    assert res.h2d_bytes_per_step is None
    assert "CPU-only" in res.transfer_measurement or "skipped" in res.transfer_measurement
    assert res.implicit_syncs is None
    assert any("CPU-only" in n for n in res.notes)
    assert res.flops_params_used == res.n_params_nonembedding
    md = res.to_markdown()
    assert "steps/sec" in md and "MFU" in md and "n/a" in md
    assert "N used in 6*N*D" in md
    assert json.dumps(res.to_dict(), default=str)


def test_benchmark_reference_params_basis() -> None:
    cfg = BenchConfig(
        steps=2, warmup=1, batch_size=2, seq_len=8, n_layers=1, d_model=16, n_heads=2,
        d_ff=32, corpus_seqs=4, target_dim=4, device="cpu", measure_transfers=False,
        params_basis="c2_c3_jepa",
    )
    res = run_benchmark(cfg, verbose=False)
    assert res.flops_params_used == REFERENCE_PARAMS["c2_c3_jepa"]
    assert any("7,408,960" in n for n in res.notes)


def test_benchmark_rejects_unknown_params_basis() -> None:
    cfg = BenchConfig(
        steps=1, warmup=1, batch_size=2, seq_len=8, n_layers=1, d_model=16, n_heads=2,
        d_ff=32, corpus_seqs=4, target_dim=4, device="cpu", measure_transfers=False,
        params_basis="nonsense",
    )
    with pytest.raises(ValueError):
        run_benchmark(cfg, verbose=False)


def test_benchmark_rejects_too_little_warmup_when_compiling() -> None:
    cfg = BenchConfig(steps=2, warmup=1, compile=True, device="cpu", corpus_seqs=4, batch_size=2)
    with pytest.raises(ValueError, match="warmup"):
        run_benchmark(cfg, verbose=False)


def test_bench_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    from xjepa.perf.bench import main

    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    code = main(
        [
            "--steps", "2", "--warmup", "1", "--batch-size", "2", "--seq-len", "8",
            "--n-layers", "1", "--d-model", "16", "--n-heads", "2", "--d-ff", "32",
            "--corpus-seqs", "4", "--device", "cpu", "--no-transfers", "--quiet",
            "--json", str(out_json), "--markdown", str(out_md),
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text())
    assert payload["steps_per_sec"] > 0
    assert "# x-JEPA training-step benchmark" in out_md.read_text()


def test_profile_report_runs_on_cpu() -> None:
    cfg = BenchConfig(
        steps=2, warmup=1, batch_size=2, seq_len=16, n_layers=2, d_model=32, n_heads=4,
        d_ff=64, corpus_seqs=8, target_dim=8, device="cpu",
    )
    report = run_profile(cfg, steps=2, warmup=1, top_k=5)
    assert report.device == "cpu"
    assert report.top_ops, "profiler recorded no operators"
    assert report.total_self_time_us > 0
    assert report.copies == [], "a CPU run cannot have host-device copies"
    md = report.to_markdown()
    assert "Top stalls" in md
    assert "Not applicable" in md  # CPU run: no host-device boundary to audit
    assert "torch.compile" in md
