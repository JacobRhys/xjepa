"""CPU tests for the GPU-resident data layer (tiny synthetic shapes).

Run from the repo root::

    python -m pytest tests/test_data.py -q
"""

from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import pytest
import torch

from xjepa.data.build_cache import build_cache, fit_pca, rankme
from xjepa.data.masking import (
    IGNORE_INDEX,
    MASK_ID,
    MaskingConfig,
    Masker,
    random_mask,
    span_mask,
)
from xjepa.data.store import PAD_ID, Batch, BucketBatcher, GpuCorpus

DEVICE = "cpu"
TARGET_DIM = 8  # tiny stand-in for the real 128


def make_corpus(
    lengths: np.ndarray, target_dim: int = TARGET_DIM, seed: int = 0
) -> Tuple[GpuCorpus, np.ndarray, np.ndarray, np.ndarray]:
    """Build a tiny synthetic corpus plus the raw numpy arrays behind it.

    Targets encode their own flat residue index so a gather can be verified
    exactly: ``targets[i, d] = i + d / 64`` (all fp16-exact for small ``i``).

    Args:
        lengths: per-sequence lengths.
        target_dim: width of the target bank.
        seed: RNG seed for token ids.

    Returns:
        ``(corpus, tokens, offsets, targets)``.
    """
    rng = np.random.default_rng(seed)
    total = int(lengths.sum())
    tokens = rng.integers(4, 24, size=total, dtype=np.uint8)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    # modulo keeps the encoded index inside fp16's exact-integer range
    idx = (np.arange(total, dtype=np.float32) % 2048.0)[:, None]
    dd = np.arange(target_dim, dtype=np.float32)[None, :] / 64.0
    targets = (idx + dd).astype(np.float16)
    corpus = GpuCorpus.from_arrays(tokens, offsets, targets, device=DEVICE)
    return corpus, tokens, offsets, targets


def realistic_lengths(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Protein-like length distribution: lognormal, median ~260, capped at 700."""
    rng = np.random.default_rng(seed)
    lens = rng.lognormal(mean=np.log(260.0), sigma=0.55, size=n)
    return np.clip(lens, 30, 700).astype(np.int32)


# --------------------------------------------------------------------- store


def test_corpus_fields_and_residency() -> None:
    """Corpus exposes contract dtypes/shapes and reports its own size."""
    lengths = np.array([5, 9, 3, 12], dtype=np.int32)
    corpus, tokens, offsets, targets = make_corpus(lengths)

    assert corpus.tokens.dtype == torch.uint8
    assert corpus.offsets.dtype == torch.int32
    assert corpus.targets.dtype == torch.float16
    assert corpus.lengths.dtype == torch.int32
    assert corpus.n_seqs == 4
    assert corpus.total_residues == int(lengths.sum())
    assert torch.equal(corpus.lengths, torch.tensor(lengths, dtype=torch.int32))

    expected = tokens.nbytes + offsets.nbytes + targets.nbytes + lengths.astype(
        np.int32
    ).nbytes
    assert corpus.nbytes == expected

    summary = corpus.summary()
    assert summary["n_seqs"] == 4
    assert summary["target_dim"] == TARGET_DIM
    assert summary["targets"]["dtype"] == "float16"
    assert summary["total_mb"] == round(expected / 1024 ** 2, 3)
    assert summary["total_gb"] == round(expected / 1024 ** 3, 4)


def test_corpus_load_roundtrip(tmp_path) -> None:
    """A memory-mapped .npy cache loads back identically."""
    lengths = np.array([7, 11, 5], dtype=np.int32)
    corpus, tokens, offsets, targets = make_corpus(lengths)
    d = str(tmp_path)
    np.save(os.path.join(d, "tokens.npy"), tokens)
    np.save(os.path.join(d, "offsets.npy"), offsets)
    np.save(os.path.join(d, "targets.npy"), targets)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"rankme": 6.5}, fh)

    loaded = GpuCorpus.load(d, device=DEVICE, target_dim=TARGET_DIM)
    assert torch.equal(loaded.tokens, corpus.tokens)
    assert torch.equal(loaded.offsets, corpus.offsets)
    assert torch.equal(loaded.targets, corpus.targets)
    assert loaded.meta["rankme"] == 6.5

    with pytest.raises(ValueError):
        GpuCorpus.load(d, device=DEVICE, target_dim=TARGET_DIM + 1)


def test_corpus_rejects_bad_offsets() -> None:
    """Offsets must start at 0 and end at total_residues."""
    tokens = torch.zeros(10, dtype=torch.uint8)
    targets = torch.zeros(10, TARGET_DIM, dtype=torch.float16)
    bad = torch.tensor([0, 4, 9], dtype=torch.int32)  # 9 != 10
    with pytest.raises(ValueError):
        GpuCorpus(tokens, bad, targets)


# -------------------------------------------------------------------- gather


def test_gather_matches_naive_python_loop() -> None:
    """The broadcast index-matrix gather equals a naive per-sequence loop."""
    lengths = np.array([5, 130, 64, 300, 12, 511, 128], dtype=np.int32)
    corpus, tokens, offsets, targets = make_corpus(lengths)
    batcher = BucketBatcher(
        corpus,
        buckets=(16, 64, 128),
        token_budget=256,
        policy="pad",
        drop_last=False,
        batch_multiple=1,
    )
    rows = torch.tensor([0, 4], dtype=torch.int64)  # lengths 5 and 12 -> bucket 16
    batch = batcher.make_batch(0, rows)
    blen = batcher.buckets[0]

    # Naive reference.
    ref_tokens = np.full((len(rows), blen), PAD_ID, dtype=np.int64)
    ref_targets = np.zeros((len(rows), blen, TARGET_DIM), dtype=np.float16)
    ref_pad = np.zeros((len(rows), blen), dtype=bool)
    for b, r in enumerate(rows.tolist()):
        start, end = int(offsets[r]), int(offsets[r + 1])
        n = min(end - start, blen)
        for t in range(n):
            ref_tokens[b, t] = int(tokens[start + t])
            ref_targets[b, t] = targets[start + t]
            ref_pad[b, t] = True

    assert np.array_equal(batch.tokens.numpy(), ref_tokens)
    assert np.array_equal(batch.pad_mask.numpy(), ref_pad)
    assert np.array_equal(batch.targets.numpy(), ref_targets)
    assert batch.bucket == blen


def test_gather_truncates_and_crops_to_contiguous_window() -> None:
    """With policy='crop' each row is a contiguous window of its sequence."""
    lengths = np.array([300, 400, 512, 260], dtype=np.int32)
    corpus, tokens, offsets, _ = make_corpus(lengths)
    batcher = BucketBatcher(
        corpus, buckets=(128, 256), token_budget=512, policy="crop", batch_multiple=1
    )
    rows = torch.arange(4, dtype=torch.int64)
    batch = batcher.make_batch(1, rows)  # bucket 256
    assert batch.tokens.shape == (4, 256)
    assert bool(batch.pad_mask.all())  # every sequence is >= 256, nothing padded

    # targets encode the flat residue index -> recover the crop start exactly.
    starts = batch.targets[:, 0, 0].to(torch.int64)  # residue index, mod 2048
    step = batch.targets[:, :, 0].to(torch.int64)
    expect = starts.unsqueeze(1) + torch.arange(256).unsqueeze(0)
    assert torch.equal(step, expect)
    for b, r in enumerate(rows.tolist()):
        s = int(starts[b])
        assert int(offsets[r]) <= s <= int(offsets[r + 1]) - 256
        window = tokens[s : s + 256].astype(np.int64)
        assert np.array_equal(batch.tokens[b].numpy(), window)


def test_padding_positions_are_zeroed_targets_and_pad_token() -> None:
    """Padding slots carry pad_id and zeroed targets."""
    lengths = np.array([3, 9], dtype=np.int32)
    corpus, _, _, _ = make_corpus(lengths)
    batcher = BucketBatcher(
        corpus, buckets=(16,), token_budget=32, policy="pad", batch_multiple=1
    )
    batch = batcher.make_batch(0, torch.tensor([0, 1], dtype=torch.int64))
    pad = ~batch.pad_mask
    assert bool((batch.tokens[pad] == PAD_ID).all())
    assert bool((batch.targets[pad] == 0).all())
    assert int(batch.pad_mask.sum()) == 12


# ------------------------------------------------------------------ batching


def test_bucket_batch_shapes_are_fixed_and_token_budget_respected() -> None:
    """Only a small set of static (B, L) shapes is ever emitted."""
    lengths = realistic_lengths(2000, seed=1)
    corpus, _, _, _ = make_corpus(lengths)
    batcher = BucketBatcher(corpus, token_budget=65_536, seed=0)
    shapes = batcher.batch_shapes()
    assert shapes == {128: (512, 128), 256: (256, 256), 384: (168, 384), 512: (128, 512)}
    for bucket, (bs, blen) in shapes.items():
        assert bs * blen <= 65_536
        assert bs * blen >= 0.9 * 65_536

    seen = set()
    n_batches = 0
    for batch in batcher:
        assert isinstance(batch, Batch)
        seen.add(tuple(batch.tokens.shape))
        assert batch.tokens.shape == (shapes[batch.bucket][0], batch.bucket)
        assert batch.targets.shape == (
            shapes[batch.bucket][0],
            batch.bucket,
            TARGET_DIM,
        )
        assert batch.tokens.dtype == torch.int64
        assert batch.targets.dtype == torch.float16
        assert batch.pad_mask.dtype == torch.bool
        n_batches += 1
    assert n_batches == len(batcher) > 0
    assert seen <= {(bs, blen) for blen, (bs, _) in shapes.items()}


def test_padding_efficiency_above_threshold() -> None:
    """Token-budget bucketing keeps padding waste under 8%."""
    lengths = realistic_lengths(4000, seed=2)
    corpus, _, _, _ = make_corpus(lengths)
    batcher = BucketBatcher(corpus, token_budget=65_536, seed=0)
    eff = batcher.padding_efficiency()
    assert 0.0 < eff <= 1.0
    assert eff > 0.92, f"padding efficiency {eff:.4f} below 0.92"

    # empirical check: measured real/padded over a whole epoch matches.
    real = 0
    padded = 0
    for batch in batcher:
        real += int(batch.pad_mask.sum())
        padded += batch.pad_mask.numel()
    assert real / padded == pytest.approx(eff, abs=0.02)

    # the naive pad-to-512 collate is much worse, as the contract claims.
    naive = float(np.minimum(lengths, 512).sum()) / (512.0 * len(lengths))
    assert naive < 0.75


def test_bucket_policies_trade_padding_against_cropping() -> None:
    """hybrid (default) keeps padding waste < 8%; pad-up alone does not."""
    lengths = realistic_lengths(4000, seed=11)
    corpus, _, _, _ = make_corpus(lengths)
    hybrid = BucketBatcher(corpus, token_budget=65_536, seed=0)  # policy="hybrid"
    crop = BucketBatcher(corpus, token_budget=65_536, seed=0, policy="crop")
    pad = BucketBatcher(corpus, token_budget=65_536, seed=0, policy="pad")

    assert hybrid.policy == "hybrid"
    assert hybrid.padding_efficiency() > 0.92
    assert crop.padding_efficiency() > hybrid.padding_efficiency() > pad.padding_efficiency()
    # ...and hybrid discards fewer residues per epoch than pure cropping.
    assert crop.crop_loss_fraction() > hybrid.crop_loss_fraction() > 0.0
    # pad-up alone cannot meet the contract's <8% waste claim.
    assert pad.padding_efficiency() < 0.92

    # every sequence lands in a bucket under every policy
    for b in (hybrid, crop, pad):
        assert sum(b.bucket_counts().values()) == len(lengths)


def test_hybrid_pads_up_when_bucket_is_nearly_full() -> None:
    """A sequence at >= pad_threshold occupancy pads up instead of cropping."""
    lengths = np.array([250, 130, 500, 300], dtype=np.int32)
    corpus, _, _, _ = make_corpus(lengths)
    b = BucketBatcher(
        corpus,
        buckets=(128, 256, 384, 512),
        token_budget=1024,
        pad_threshold=0.9,
        batch_multiple=1,
    )
    counts = b.bucket_counts()
    # 250/256 = 0.977 -> pads into 256; 500/512 = 0.977 -> pads into 512
    # 130 -> 130/256 = 0.51 -> crops into 128; 300 -> 300/384 = 0.78 -> crops to 256
    assert counts == {128: 1, 256: 2, 384: 0, 512: 1}


def test_every_sequence_used_once_without_drop_last() -> None:
    """With drop_last=False the epoch covers each kept sequence exactly once."""
    lengths = realistic_lengths(500, seed=3)
    corpus, _, _, _ = make_corpus(lengths)
    batcher = BucketBatcher(
        corpus, token_budget=4096, seed=0, drop_last=False, batch_multiple=1
    )
    assert sum(batcher.bucket_counts().values()) == len(lengths)
    total = sum(int(b.tokens.shape[0]) for b in batcher)
    assert total == len(lengths)


def test_epoch_permutation_changes_but_is_seeded() -> None:
    """Two batchers with the same seed emit identical epochs."""
    lengths = realistic_lengths(300, seed=4)
    corpus, _, _, _ = make_corpus(lengths)
    a = BucketBatcher(corpus, token_budget=2048, seed=7, policy="pad")
    b = BucketBatcher(corpus, token_budget=2048, seed=7, policy="pad")
    ba = [x.tokens for x in a]
    bb = [x.tokens for x in b]
    assert len(ba) == len(bb) and len(ba) > 1
    assert all(torch.equal(x, y) for x, y in zip(ba, bb))
    second = [x.tokens for x in a]
    assert not all(
        torch.equal(x, y) for x, y in zip(ba, second) if x.shape == y.shape
    )


# ------------------------------------------------------------------- masking


def _pad_mask(b: int, length: int, lens: list[int]) -> torch.Tensor:
    ar = torch.arange(length).unsqueeze(0)
    return ar < torch.tensor(lens).unsqueeze(1)


@pytest.mark.parametrize("mode", ["random", "span"])
def test_padding_is_never_masked(mode: str) -> None:
    """mask_sel is always a subset of pad_mask."""
    pad_mask = _pad_mask(16, 512, [512, 400, 9, 1] * 4)
    tokens = torch.randint(4, 24, pad_mask.shape, dtype=torch.int64)
    masker = Masker(MaskingConfig(mode=mode), device=DEVICE, seed=0)
    out, mask_sel, labels = masker(tokens, pad_mask)
    assert not bool((mask_sel & ~pad_mask).any())
    assert bool((labels[~pad_mask] == IGNORE_INDEX).all())
    assert bool((out[~pad_mask] == tokens[~pad_mask]).all())


@pytest.mark.parametrize("mode", ["random", "span"])
def test_mask_rate_within_tolerance(mode: str) -> None:
    """Realised mask rate over real residues is close to 15%."""
    pad_mask = _pad_mask(64, 512, [512] * 32 + [300] * 32)
    masker = Masker(MaskingConfig(mode=mode, rate=0.15), device=DEVICE, seed=1)
    mask_sel = masker.select(pad_mask)
    rate = float(mask_sel.sum()) / float(pad_mask.sum())
    assert 0.12 < rate < 0.18, f"{mode} rate {rate:.4f}"


def test_labels_and_mlm_corruption_split() -> None:
    """ESM-2 80/10/10: labels hold originals, corruption respects the split."""
    pad_mask = torch.ones(128, 256, dtype=torch.bool)
    tokens = torch.randint(4, 24, pad_mask.shape, dtype=torch.int64)
    masker = Masker(
        MaskingConfig(mode="random", rate=0.15, corruption="mlm"), device=DEVICE, seed=2
    )
    out, mask_sel, labels = masker(tokens, pad_mask)

    assert bool((labels[mask_sel] == tokens[mask_sel]).all())
    assert bool((labels[~mask_sel] == IGNORE_INDEX).all())
    assert bool((out[~mask_sel] == tokens[~mask_sel]).all())

    n = int(mask_sel.sum())
    masked_out = out[mask_sel]
    orig = tokens[mask_sel]
    frac_mask = float((masked_out == MASK_ID).sum()) / n
    kept_or_rand = masked_out != MASK_ID
    frac_same = float((kept_or_rand & (masked_out == orig)).sum()) / n
    assert 0.76 < frac_mask < 0.84
    # 10% kept unchanged, plus the ~1/20 of the random branch that redraws the
    # original amino acid.
    assert 0.07 < frac_same < 0.14
    assert bool(((masked_out >= 4) & (masked_out <= 32)).all())


def test_jepa_corruption_uses_mask_token_everywhere() -> None:
    """On the JEPA path every masked position becomes <mask>."""
    pad_mask = _pad_mask(8, 128, [128, 100, 64, 33] * 2)
    tokens = torch.randint(4, 24, pad_mask.shape, dtype=torch.int64)
    masker = Masker(
        MaskingConfig(mode="span", corruption="jepa"), device=DEVICE, seed=3
    )
    out, mask_sel, _ = masker(tokens, pad_mask)
    assert bool((out[mask_sel] == MASK_ID).all())
    assert bool((out[~mask_sel] == tokens[~mask_sel]).all())


def test_span_length_distribution_is_sane() -> None:
    """Span runs have mean length near mean_span and are contiguous."""
    pad_mask = torch.ones(256, 512, dtype=torch.bool)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(5)
    mask_sel = span_mask(pad_mask, rate=0.15, mean_span=8.0, generator=gen)

    m = mask_sel.to(torch.int8)
    starts = int((m[:, 1:] > m[:, :-1]).sum()) + int(m[:, 0].sum())
    mean_run = float(m.sum()) / starts
    assert 5.0 < mean_run < 14.0, f"mean run {mean_run:.2f}"

    # random masking must produce much shorter runs than span masking
    rnd = random_mask(pad_mask, 0.15, gen).to(torch.int8)
    r_starts = int((rnd[:, 1:] > rnd[:, :-1]).sum()) + int(rnd[:, 0].sum())
    assert float(rnd.sum()) / r_starts < 1.5 < mean_run


def test_every_row_masks_at_least_one_position() -> None:
    """Even a length-1 sequence gets a masked position (and it is not padding)."""
    pad_mask = _pad_mask(32, 64, [1] * 32)
    for mode in ("random", "span"):
        masker = Masker(MaskingConfig(mode=mode), device=DEVICE, seed=6)
        mask_sel = masker.select(pad_mask)
        assert bool(mask_sel.any(dim=1).all())
        assert not bool((mask_sel & ~pad_mask).any())


def test_batcher_uses_masker_and_never_masks_padding() -> None:
    """End-to-end: batches carry a contract-shaped mask_sel / labels."""
    lengths = realistic_lengths(600, seed=7)
    corpus, _, _, _ = make_corpus(lengths)
    masker = Masker(MaskingConfig(mode="span", corruption="jepa"), device=DEVICE, seed=8)
    batcher = BucketBatcher(corpus, token_budget=8192, masker=masker, seed=0)
    n = 0
    for batch in batcher:
        assert batch.mask_sel.shape == batch.tokens.shape
        assert batch.labels.dtype == torch.int64
        assert not bool((batch.mask_sel & ~batch.pad_mask).any())
        assert bool((batch.labels[~batch.mask_sel] == IGNORE_INDEX).all())
        n += 1
        if n == 5:
            break
    assert n == 5


# ---------------------------------------------------------------- build_cache


def test_fit_pca_and_rankme_on_synthetic_low_rank_data() -> None:
    """PCA recovers a planted low-rank structure; RankMe reflects it."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(2000, 6)).astype(np.float32)
    mix = rng.normal(size=(6, 32)).astype(np.float32)
    x = latent @ mix + 0.01 * rng.normal(size=(2000, 32)).astype(np.float32)

    fit = fit_pca(x, dim=6, standardise=True)
    assert fit.components.shape == (32, 6)
    assert fit.explained_variance_ratio > 0.99
    proj = (torch.as_tensor(x) - fit.mean) @ fit.components / fit.scale
    assert float(proj.std(dim=0).mean()) == pytest.approx(1.0, abs=0.05)
    rm = rankme(proj)
    assert 3.0 < rm <= 6.5


def test_build_cache_end_to_end(tmp_path) -> None:
    """The offline builder writes a loadable fp16 cache and reports metrics."""
    rng = np.random.default_rng(1)
    n, d_in, d_out = 1000, 32, 8
    latent = rng.normal(size=(n, 12)).astype(np.float32)
    emb = (latent @ rng.normal(size=(12, d_in)).astype(np.float32)).astype(np.float32)
    lengths = np.array([200, 300, 250, 250], dtype=np.int32)
    offsets = np.zeros(5, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    tokens = rng.integers(4, 24, size=n, dtype=np.uint8)

    raw = tmp_path / "raw"
    raw.mkdir()
    np.save(raw / "emb.npy", emb)
    np.save(raw / "tokens.npy", tokens)
    np.save(raw / "offsets.npy", offsets)
    out_dir = str(tmp_path / "cache")

    meta = build_cache(
        embeddings=str(raw / "emb.npy"),
        out_dir=out_dir,
        dim=d_out,
        tokens=str(raw / "tokens.npy"),
        offsets=str(raw / "offsets.npy"),
        subsample=500,
        chunk_rows=128,
        rankme_rows=500,
        seed=0,
    )
    assert meta["target_dim"] == d_out
    assert 0.0 < meta["explained_variance_ratio"] <= 1.0
    assert 1.0 < meta["rankme"] <= d_out + 1e-6
    assert meta["target_bank_bytes"] == n * d_out * 2

    corpus = GpuCorpus.load(out_dir, device=DEVICE, target_dim=d_out)
    assert corpus.targets.shape == (n, d_out)
    assert corpus.targets.dtype == torch.float16
    assert corpus.n_seqs == 4
    assert corpus.meta["rankme"] == meta["rankme"]


# ------------------------------------------------------------------ policies


def test_bucket_policies_tradeoff() -> None:
    """hybrid (default) beats pad on waste and beats crop on residues kept."""
    lengths = realistic_lengths(4000, seed=11)
    corpus, _, _, _ = make_corpus(lengths)
    kw = dict(token_budget=65_536, seed=0)
    hyb = BucketBatcher(corpus, policy="hybrid", **kw)
    crop = BucketBatcher(corpus, policy="crop", **kw)
    pad = BucketBatcher(corpus, policy="pad", **kw)

    assert hyb.policy == BucketBatcher(corpus, **kw).policy == "hybrid"
    assert hyb.padding_efficiency() > 0.92 > pad.padding_efficiency()
    assert crop.padding_efficiency() > hyb.padding_efficiency()
    assert hyb.crop_loss_fraction() < crop.crop_loss_fraction()
    assert pad.crop_loss_fraction() <= hyb.crop_loss_fraction()


def test_batches_stay_on_corpus_device() -> None:
    """Nothing leaves the corpus device during batching."""
    lengths = realistic_lengths(200, seed=12)
    corpus, _, _, _ = make_corpus(lengths)
    masker = Masker(MaskingConfig(), device=DEVICE, seed=0)
    batcher = BucketBatcher(corpus, token_budget=2048, masker=masker, seed=0)
    batch = next(iter(batcher))
    for t in (batch.tokens, batch.targets, batch.pad_mask, batch.mask_sel, batch.labels):
        assert t.device == corpus.device
