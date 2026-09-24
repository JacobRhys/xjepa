"""Cross-module integration: build every condition from the REAL modules.

Each agent's own test file stubs the other two layers, so a signature drift
between `xjepa.model.heads` and `xjepa.train.trainer.build_model` cannot show up
there. This file wires the genuine encoder, heads, masker and objectives
together and takes one optimiser step per condition on CPU.
"""

from __future__ import annotations

import pytest
import torch
from dataclasses import replace

from xjepa.data.masking import MaskingConfig, Masker
from xjepa.data.store import Batch
from xjepa.train.objectives import OBJECTIVE_NAMES, ObjectiveConfig, build_objective
from xjepa.train.trainer import RunConfig, build_model, build_optimizer

TARGET_DIM = 128
B, L, VOCAB = 3, 64, 33


def _tiny_config(name: str) -> RunConfig:
    """A 2-layer stand-in for the 6-layer run config; same wiring, fast on CPU."""
    cfg = RunConfig(
        name=name,
        device="cpu",
        compile=False,
        target_dim=TARGET_DIM,
        objective=replace(ObjectiveConfig(), name=name),
    )
    cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff = 2, 64, 4, 128
    cfg.max_len, cfg.vocab = L, VOCAB
    return cfg


def _batch(uses_masking: bool, corruption: str) -> Batch:
    g = torch.Generator(device="cpu").manual_seed(0)
    tokens = torch.randint(4, VOCAB, (B, L), generator=g)
    pad_mask = torch.ones(B, L, dtype=torch.bool)
    pad_mask[1, L // 2 :] = False  # one genuinely ragged row
    pad_mask[2, -3:] = False
    tokens = torch.where(pad_mask, tokens, torch.ones_like(tokens))

    if uses_masking:
        masker = Masker(MaskingConfig(rate=0.15, corruption=corruption), generator=g)
        tokens_out, mask_sel, labels = masker(tokens, pad_mask)
    else:
        tokens_out = tokens
        mask_sel = torch.zeros(B, L, dtype=torch.bool)
        labels = torch.full((B, L), -100)

    targets = torch.randn(B, L, TARGET_DIM, generator=g).to(torch.float16)
    return Batch(
        tokens=tokens_out,
        targets=targets,
        pad_mask=pad_mask,
        mask_sel=mask_sel,
        labels=labels,
        bucket=L,
    )


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_condition_builds_and_steps(name: str) -> None:
    """build_model must accept the real head constructors, and a step must run."""
    cfg = _tiny_config(name)
    objective = build_objective(cfg.objective)
    model = build_model(cfg, objective)
    opt = build_optimizer(model, cfg)

    corruption = "mlm" if objective.needs_mlm_head else "jepa"
    batch = _batch(objective.uses_masking, corruption)

    loss, metrics = objective.loss(model, batch)
    assert loss.ndim == 0 and torch.isfinite(loss), f"{name}: bad loss {loss}"
    for k, v in metrics.items():
        assert isinstance(v, torch.Tensor), f"{name}: metric {k} synced to host"

    loss.backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, f"{name}: no gradient reached the encoder"
    assert any(g.abs().sum() > 0 for g in grads), f"{name}: encoder grads all zero"
    opt.step()
    opt.zero_grad(set_to_none=True)


@pytest.mark.parametrize("name", ["c5_distil", "c5b_masked_distil"])
def test_mechanism_controls_have_no_predictor(name: str) -> None:
    """C5/C5b must not merely bypass the predictor -- it must not exist."""
    cfg = _tiny_config(name)
    model = build_model(cfg, build_objective(cfg.objective))
    assert model.predictor is None, f"{name} constructed a predictor"


def test_mlm_labels_survive_all_three_corruption_branches() -> None:
    """ESM-2 corruption keeps 10% and randomises 10%; labels must still be clean.

    C5/C5c reconstruct the unmasked sequence from `labels`, so a label set only
    at the 80% [MASK] branch would silently feed them corrupted tokens.
    """
    g = torch.Generator(device="cpu").manual_seed(7)
    tokens = torch.randint(4, VOCAB, (16, L), generator=g)
    pad_mask = torch.ones(16, L, dtype=torch.bool)
    masker = Masker(MaskingConfig(rate=0.5, corruption="mlm"), generator=g)
    out, mask_sel, labels = masker(tokens, pad_mask)

    assert torch.equal(labels[mask_sel], tokens[mask_sel])
    assert (labels[~mask_sel] == -100).all()
    # with rate 0.5 the 10%-keep and 10%-random branches are both exercised
    changed = (out != tokens) & mask_sel
    assert changed.any() and (~changed & mask_sel).any()

    recovered = torch.where(labels >= 0, labels, out)
    assert torch.equal(recovered, tokens), "unmasked view is not recoverable"


def test_encoder_accepts_no_pad_mask() -> None:
    """`pad_mask=None` is the crop policy's zero-padding path.

    It must be numerically identical to passing an all-true mask -- the point of
    `None` is purely that it lets SDPA reach the flash backend, not that it
    computes something different.
    """
    from xjepa.model.encoder import Encoder, EncoderConfig

    torch.manual_seed(0)
    enc = Encoder(EncoderConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128, max_len=L))
    enc.eval()
    tokens = torch.randint(4, VOCAB, (2, L))
    all_true = torch.ones(2, L, dtype=torch.bool)

    with torch.no_grad():
        masked = enc(tokens, all_true)
        unmasked = enc(tokens, None)

    torch.testing.assert_close(masked, unmasked, rtol=1e-4, atol=1e-5)


def test_pilot_costing_is_monotone_in_throughput() -> None:
    """Faster measured throughput must never cost more."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.pilot import cost_from_throughput

    slow = cost_from_throughput("slow", 1.0e6, 0.18, rate=0.34)
    fast = cost_from_throughput("fast", 1.4e6, 0.25, rate=0.34)
    assert fast.total_gbp < slow.total_gbp
    assert fast.minutes_per_run < slow.minutes_per_run
    # 1.4e6 tok/s -> ~12 min/run, 23 runs -> the plan's ~£6 envelope
    assert fast.within_budget


def _tiny_corpus(tmp_path, n_seqs: int = 64, dim: int = 8):
    """Write a minimal on-disk corpus in the layout GpuCorpus.load expects."""
    import numpy as np

    rng = np.random.default_rng(0)
    lengths = rng.integers(20, 60, size=n_seqs)
    offsets = np.zeros(n_seqs + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])

    d = tmp_path / "corpus"
    d.mkdir()
    np.save(d / "tokens.npy", rng.integers(4, 24, size=total).astype("uint8"))
    np.save(d / "offsets.npy", offsets.astype("int32"))
    np.save(d / "targets.npy", rng.standard_normal((total, dim)).astype("float16"))
    return d


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_build_batches_wires_to_the_real_batcher(tmp_path, name: str) -> None:
    """build_batches must match BucketBatcher/Masker's actual signatures.

    Nothing else covers this: the trainer's own tests feed hand-built batches,
    so a keyword-argument drift here surfaces only when a real corpus is loaded
    -- i.e. on the rented GPU, after paying for it.
    """
    from xjepa.train.trainer import build_batches

    corpus_dir = _tiny_corpus(tmp_path)
    cfg = _tiny_config(name)
    cfg.corpus_path = str(corpus_dir)
    cfg.target_dim = 8
    cfg.buckets = (32, 64)
    cfg.tokens_per_step = 512
    objective = build_objective(cfg.objective)

    corpus, batcher = build_batches(cfg, objective)
    batch = next(iter(batcher))

    assert batch.tokens.shape == batch.pad_mask.shape
    assert batch.targets.shape[-1] == 8
    assert batch.bucket in cfg.buckets

    if objective.uses_masking:
        assert batch.mask_sel.any(), f"{name}: masker produced no masked positions"
        assert not (batch.mask_sel & ~batch.pad_mask).any(), "padding was masked"
    else:
        # No masker: labels are all -100, so the clean sequence is recoverable.
        assert not batch.mask_sel.any()
        assert (batch.labels == -100).all()


def test_build_batches_corruption_follows_the_objective(tmp_path) -> None:
    """MLM conditions get 80/10/10; latent conditions get pure <mask>."""
    from xjepa.data.masking import MASK_ID
    from xjepa.train.trainer import build_batches

    corpus_dir = _tiny_corpus(tmp_path)

    def batch_for(name: str):
        cfg = _tiny_config(name)
        cfg.corpus_path, cfg.target_dim = str(corpus_dir), 8
        cfg.buckets, cfg.tokens_per_step = (32, 64), 512
        return next(iter(build_batches(cfg, build_objective(cfg.objective))[1]))

    jepa = batch_for("c3_jepa_frozen")
    assert (jepa.tokens[jepa.mask_sel] == MASK_ID).all(), "JEPA must mask every selected position"

    mlm = batch_for("c1_mlm")
    masked_inputs = mlm.tokens[mlm.mask_sel]
    assert (masked_inputs != MASK_ID).any(), "MLM should leave ~20% un-masked (80/10/10)"
