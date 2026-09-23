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
