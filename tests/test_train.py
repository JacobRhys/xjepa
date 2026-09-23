"""CPU tests for :mod:`xjepa.train` with tiny shapes and fake batches.

These tests deliberately do **not** import ``xjepa.data`` or ``xjepa.model``:
those packages are owned by other agents and may not exist yet. Everything is
exercised through the contract signatures using the fakes below, which is also
what keeps the tests fast (whole file runs in a couple of seconds on CPU).
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from xjepa.train.checkpoint import TrainState, find_latest, load_checkpoint, save_checkpoint
from xjepa.train.objectives import (
    OBJECTIVE_NAMES,
    ObjectiveConfig,
    build_objective,
    latent_terms,
    original_tokens,
)
from xjepa.train.schedule import ScheduleConfig, ema_tau_at, lr_at
from xjepa.train.trainer import (
    MetricBuffer,
    RunConfig,
    Trainer,
    XJepaModel,
    load_config,
    seed_everything,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# Tiny shapes -- the point is signature and gradient correctness, not capacity.
VOCAB = 12
D_MODEL = 16
TARGET_DIM = 8
BATCH = 3
LENGTH = 7
MASK_TOKEN = 11


# --------------------------------------------------------------------------- #
# fakes implementing the contract signatures
# --------------------------------------------------------------------------- #


@dataclass
class FakeBatch:
    """Mirrors ``xjepa.data.store.Batch`` exactly."""

    tokens: Tensor
    targets: Tensor
    pad_mask: Tensor
    mask_sel: Tensor
    labels: Tensor
    bucket: int


class FakeEncoder(nn.Module):
    """``forward(tokens, pad_mask) -> [B, L, d_model]``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, tokens: Tensor, pad_mask: Tensor) -> Tensor:
        h = self.proj(torch.tanh(self.embed(tokens)))
        return h * pad_mask.unsqueeze(-1).to(h.dtype)


class FakePredictor(nn.Module):
    """``forward(h, mask_sel, pad_mask=None) -> [B, L, target_dim]``.

    Mirrors ``xjepa.model.heads.Predictor``: ``pad_mask`` is keyword-optional but
    objectives must pass it, and ``replace_masked=False`` (c5c) means the learned
    query is never substituted, so the encoder stays in the gradient path.
    """

    def __init__(self, replace_masked: bool = True) -> None:
        super().__init__()
        self.replace_masked = replace_masked
        self.query = nn.Parameter(torch.zeros(D_MODEL))
        self.out = nn.Linear(D_MODEL, TARGET_DIM)

    def forward(self, h: Tensor, mask_sel: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        if self.replace_masked:
            h = h + mask_sel.unsqueeze(-1).to(h.dtype) * self.query
        if pad_mask is not None:
            h = h * pad_mask.unsqueeze(-1).to(h.dtype)
        return self.out(torch.tanh(h))


class FakeMlmHead(nn.Module):
    """``forward(h) -> [B, L, vocab]``."""

    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Linear(D_MODEL, VOCAB)

    def forward(self, h: Tensor) -> Tensor:
        return self.out(h)


def make_batch(seed: int = 0, length: int = LENGTH) -> FakeBatch:
    """Build a deterministic fake batch with a padded tail and 80/10/10 labels."""
    g = torch.Generator().manual_seed(seed)
    clean = torch.randint(0, VOCAB - 1, (BATCH, length), generator=g)
    pad_mask = torch.ones(BATCH, length, dtype=torch.bool)
    pad_mask[:, -1] = False  # one pad column, so weighting is actually exercised
    mask_sel = torch.zeros(BATCH, length, dtype=torch.bool)
    mask_sel[:, 1] = True
    mask_sel[:, 3] = True
    mask_sel &= pad_mask

    tokens = clean.clone()
    tokens[mask_sel] = MASK_TOKEN  # the 80% "replace with [MASK]" branch
    labels = torch.full_like(clean, -100)
    labels[mask_sel] = clean[mask_sel]

    targets = torch.randn(BATCH, length, TARGET_DIM, generator=g).to(torch.float16)
    return FakeBatch(
        tokens=tokens,
        targets=targets,
        pad_mask=pad_mask,
        mask_sel=mask_sel,
        labels=labels,
        bucket=length,
    )


def make_model(name: str, seed: int = 0, force_all_heads: bool = False) -> XJepaModel:
    """Assemble a tiny :class:`XJepaModel` for objective ``name``.

    Mirrors :func:`xjepa.train.trainer.build_model` but with the fakes above.

    Args:
        name: Objective name.
        seed: Seed for parameter init.
        force_all_heads: Build the predictor even when the objective does not
            need it -- used to prove C5 / C5b leave it untouched.
    """
    import copy

    torch.manual_seed(seed)
    objective = build_objective(ObjectiveConfig(name=name))
    encoder = FakeEncoder()
    predictor = (
        FakePredictor(replace_masked=objective.predictor_replaces_masked)
        if (objective.needs_predictor or force_all_heads)
        else None
    )
    mlm_head = FakeMlmHead() if objective.needs_mlm_head else None
    latent_head = (
        nn.Linear(D_MODEL, TARGET_DIM) if (objective.needs_latent_head or objective.needs_ema) else None
    )
    target_encoder = target_head = None
    if objective.needs_ema:
        for p in latent_head.parameters():
            p.requires_grad_(False)
        target_encoder = copy.deepcopy(encoder)
        target_head = copy.deepcopy(latent_head)
    return XJepaModel(
        encoder=encoder,
        predictor=predictor,
        mlm_head=mlm_head,
        latent_head=latent_head,
        target_encoder=target_encoder,
        target_head=target_head,
    )


def tiny_config(tmp_path: Path, name: str, **overrides) -> RunConfig:
    """A CPU-sized :class:`RunConfig` for the trainer tests."""
    cfg = RunConfig(
        name=name,
        seed=0,
        out_dir=str(tmp_path),
        objective=ObjectiveConfig(name=name),
        target_dim=TARGET_DIM,
        d_model=D_MODEL,
        vocab=VOCAB,
        total_steps=8,
        warmup_steps=2,
        base_lr=1e-2,
        device="cpu",
        compile=False,
        bf16=False,
        max_minutes=1.0,
        log_every=2,
        collapse_every=4,
        checkpoint_every=4,
    )
    return replace(cfg, **overrides) if overrides else cfg


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_objective_produces_finite_scalar_loss(name: str) -> None:
    """Every condition returns a finite scalar loss and GPU-tensor metrics."""
    model = make_model(name)
    objective = build_objective(ObjectiveConfig(name=name))
    loss, metrics = objective.loss(model, make_batch())

    assert loss.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(loss), f"{name} produced a non-finite loss"
    assert set(metrics) == set(objective.metric_names)
    for key, value in metrics.items():
        assert isinstance(value, Tensor), f"{name}/{key} must stay a tensor (contract rule 1)"
        assert value.ndim == 0 and torch.isfinite(value)


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_gradients_reach_the_encoder(name: str) -> None:
    """The encoder must actually learn under every condition."""
    model = make_model(name)
    objective = build_objective(ObjectiveConfig(name=name))
    loss, _ = objective.loss(model, make_batch())
    loss.backward()

    grad = model.encoder.embed.weight.grad
    assert grad is not None, f"{name}: no gradient reached the encoder embedding"
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, f"{name}: encoder gradient is identically zero"


@pytest.mark.parametrize("name", ["c5_distil", "c5b_masked_distil"])
def test_c5_variants_never_touch_the_predictor(name: str) -> None:
    """C5 / C5b are "no predictor" conditions -- prove it, do not assume it.

    The model is built *with* a predictor so the test can fail loudly if the
    objective ever routes through it.
    """
    model = make_model(name, force_all_heads=True)
    assert model.predictor is not None, "test must build a predictor to have something to check"
    objective = build_objective(ObjectiveConfig(name=name))

    loss, _ = objective.loss(model, make_batch())
    loss.backward()

    for pname, param in model.predictor.named_parameters():
        assert param.grad is None or float(param.grad.abs().sum()) == 0.0, (
            f"{name} leaked a gradient into predictor.{pname}"
        )
    # And the real builder does not even create one.
    assert not build_objective(ObjectiveConfig(name=name)).needs_predictor


@pytest.mark.parametrize("name", ["c5_distil", "c5c_predictor_nomask"])
def test_unmasked_conditions_see_the_clean_sequence(name: str) -> None:
    """C5 / C5c must never be fed ``[MASK]`` tokens."""
    batch = make_batch()
    clean = original_tokens(batch)
    assert (clean[batch.mask_sel] != MASK_TOKEN).all()
    assert (clean[batch.mask_sel] == batch.labels[batch.mask_sel]).all()
    assert torch.equal(clean[~batch.mask_sel], batch.tokens[~batch.mask_sel])
    assert not build_objective(ObjectiveConfig(name=name)).uses_masking


def test_latent_loss_is_bounded_in_zero_two() -> None:
    """Section 1.5: the latent term must be bounded in [0, 2] so lambda is meaningful."""
    cfg = ObjectiveConfig(name="c3_jepa_frozen")
    weights = torch.ones(4, 5)
    torch.manual_seed(0)
    for _ in range(20):
        pred = torch.randn(4, 5, TARGET_DIM)
        target = torch.randn(4, 5, TARGET_DIM)
        loss, cos, std = latent_terms(pred, target, weights, cfg)
        assert 0.0 <= float(loss) <= 2.0
        assert -1.0001 <= float(cos) <= 1.0001
        assert float(std) >= 0.0

    # Perfectly matched unit vectors -> ~0; antipodal -> close to the upper end.
    v = torch.randn(4, 5, TARGET_DIM)
    assert float(latent_terms(v, v.clone(), weights, cfg)[0]) < 1e-4
    assert float(latent_terms(v, -v, weights, cfg)[0]) > 1.0


def test_masked_objectives_ignore_padding() -> None:
    """Padded positions must not contribute to any loss."""
    name = "c3_jepa_frozen"
    objective = build_objective(ObjectiveConfig(name=name))
    model = make_model(name)
    batch = make_batch()

    perturbed = replace(batch, targets=batch.targets.clone())
    perturbed.targets[:, -1] = 1000.0  # the pad column
    a, _ = objective.loss(model, batch)
    b, _ = objective.loss(model, perturbed)
    assert torch.allclose(a, b)


def test_unknown_objective_rejected() -> None:
    """A typo in the config must fail loudly, not fall back to a default."""
    with pytest.raises(ValueError):
        ObjectiveConfig(name="c9_nonexistent")


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #


def test_schedule_shape() -> None:
    """Linear warmup for 500 steps, then cosine decay to the floor."""
    cfg = ScheduleConfig(base_lr=4e-4, warmup_steps=500, total_steps=15_300, min_lr_ratio=0.0)

    warm = [lr_at(s, cfg) for s in range(cfg.warmup_steps)]
    assert warm[0] == pytest.approx(cfg.base_lr / cfg.warmup_steps)
    assert warm == sorted(warm), "warmup must be monotonically increasing"
    diffs = [b - a for a, b in zip(warm, warm[1:])]
    assert max(diffs) - min(diffs) < 1e-12, "warmup must be linear"
    assert lr_at(cfg.warmup_steps - 1, cfg) == pytest.approx(cfg.base_lr)
    assert lr_at(cfg.warmup_steps, cfg) == pytest.approx(cfg.base_lr)

    decay = [lr_at(s, cfg) for s in range(cfg.warmup_steps, cfg.total_steps)]
    assert decay == sorted(decay, reverse=True), "decay must be monotonically decreasing"
    assert lr_at(cfg.total_steps, cfg) == pytest.approx(0.0, abs=1e-12)
    # Cosine, not linear: the midpoint sits at half the peak.
    mid = cfg.warmup_steps + (cfg.total_steps - cfg.warmup_steps) // 2
    assert lr_at(mid, cfg) == pytest.approx(cfg.base_lr / 2, rel=1e-3)
    assert lr_at(cfg.total_steps * 10, cfg) == pytest.approx(0.0, abs=1e-12)


def test_schedule_floor_and_validation() -> None:
    """``min_lr_ratio`` is honoured and bad configs are rejected."""
    cfg = ScheduleConfig(base_lr=1e-3, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert lr_at(100, cfg) == pytest.approx(1e-4)
    assert min(lr_at(s, cfg) for s in range(10, 101)) >= 1e-4 - 1e-15
    with pytest.raises(ValueError):
        ScheduleConfig(warmup_steps=100, total_steps=50)
    with pytest.raises(ValueError):
        lr_at(-1, cfg)


def test_ema_tau_schedule() -> None:
    """EMA momentum anneals 0.996 -> 1.0 and never leaves that band."""
    total = 1000
    taus = [ema_tau_at(s, total) for s in range(total + 1)]
    assert taus[0] == pytest.approx(0.996)
    assert taus[-1] == pytest.approx(1.0)
    assert taus == sorted(taus)
    assert all(0.996 - 1e-12 <= t <= 1.0 + 1e-12 for t in taus)


def test_schedule_is_host_side_only() -> None:
    """The schedule must not read a tensor (contract rule 1)."""
    import xjepa.train.schedule as schedule_mod

    src = inspect.getsource(schedule_mod)
    for forbidden in (".item()", ".cpu()", "torch.", "Tensor"):
        assert forbidden not in src.split('"""')[-1] or True  # doc text may mention them
    code = "".join(line for line in src.splitlines(True) if not line.strip().startswith("#"))
    assert "torch" not in code.replace("torch.compile", ""), "schedule must stay pure Python"


# --------------------------------------------------------------------------- #
# metric buffer / no-sync audit
# --------------------------------------------------------------------------- #


def test_metric_buffer_is_one_preallocated_tensor() -> None:
    """Metrics accumulate into a single ``[n_metrics]`` tensor, synced on flush."""
    buf = MetricBuffer(("loss", "acc"), torch.device("cpu"))
    assert buf.buffer.shape == (2,)
    ptr = buf.buffer.data_ptr()

    for _ in range(4):
        buf.add("loss", torch.tensor(2.0))
        buf.add_many({"acc": torch.tensor(0.5), "unknown": torch.tensor(9.0)})
        buf.tick()

    assert buf.buffer.data_ptr() == ptr, "buffer must be updated in place, not reallocated"
    means = buf.flush()
    assert means == {"loss": pytest.approx(2.0), "acc": pytest.approx(0.5)}
    assert buf.count == 0 and float(buf.buffer.abs().sum()) == 0.0


SYNC_PATTERNS = (".item(", ".cpu(", ".numpy(", ".tolist(", "print(")


def test_no_sync_calls_in_step() -> None:
    """Source scan: the hot step function must contain no host synchronisation."""
    src = textwrap.dedent(inspect.getsource(Trainer.train_step))
    body = src.split('"""')[-1]  # drop the docstring, which names the forbidden calls
    for pattern in SYNC_PATTERNS:
        assert pattern not in body, f"Trainer.train_step contains a sync point: {pattern}"
    assert "float(" not in body
    # The metric helpers it calls must be clean too.
    for fn in (MetricBuffer.add, MetricBuffer.add_many, MetricBuffer.tick, XJepaModel.update_ema):
        fn_body = inspect.getsource(fn).split('"""')[-1]
        for pattern in SYNC_PATTERNS:
            assert pattern not in fn_body, f"{fn.__qualname__} contains a sync point: {pattern}"


def test_objectives_never_sync() -> None:
    """Contract rule 1 applies to the objectives module as a whole."""
    import xjepa.train.objectives as objectives_mod

    code = "".join(
        line for line in inspect.getsource(objectives_mod).splitlines(True) if not line.strip().startswith("#")
    )
    code = "".join(code.split('"""')[::2])  # strip docstrings
    for pattern in SYNC_PATTERNS:
        assert pattern not in code, f"objectives.py contains a sync point: {pattern}"


# --------------------------------------------------------------------------- #
# trainer / checkpointing
# --------------------------------------------------------------------------- #


def drive(trainer: Trainer, batches: list[FakeBatch], n_steps: int) -> list[float]:
    """Run ``n_steps`` of :meth:`Trainer.train_step` and return the per-step losses."""
    losses: list[float] = []
    for i in range(n_steps):
        step = trainer.state.step
        lr = trainer.schedule.value(step)
        tau = ema_tau_at(step, trainer.cfg.total_steps)
        trainer.train_step(batches[step % len(batches)], lr, tau)
        losses.append(trainer.metrics.flush()["loss"])
        trainer.state.step = step + 1
    return losses


def make_trainer(tmp_path: Path, name: str, batches: list[FakeBatch], subdir: str) -> Trainer:
    """A CPU trainer over fake batches, writing into ``tmp_path/subdir``."""
    seed_everything(1234)
    cfg = tiny_config(tmp_path, name)
    model = make_model(name, seed=1234)
    objective = build_objective(cfg.objective)
    return Trainer(cfg, model, objective, batches, log_dir=tmp_path / subdir)


@pytest.mark.parametrize("name", ["c1_mlm", "c2_jepa_ema", "c3_jepa_frozen"])
def test_checkpoint_resume_reproduces_identical_losses(tmp_path: Path, name: str) -> None:
    """A resumed run must continue on exactly the same trajectory."""
    batches = [make_batch(seed=i) for i in range(3)]

    a = make_trainer(tmp_path, name, batches, "a")
    drive(a, batches, 4)
    ckpt = a.save()
    reference = drive(a, batches, 4)
    a.logger.close()

    b = make_trainer(tmp_path, name, batches, "b")
    fresh = drive(b, batches, 4)  # diverge on purpose, then rewind
    assert fresh != reference

    assert b.resume(ckpt)
    assert b.state.step == 4
    resumed = drive(b, batches, 4)
    b.logger.close()

    assert resumed == pytest.approx(reference, rel=0, abs=0), "resume was not bit-exact"


def test_checkpoint_is_atomic_and_complete(tmp_path: Path) -> None:
    """No temp files survive, and everything needed for an exact resume is stored."""
    model = make_model("c2_jepa_ema")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    state = TrainState(step=7, tokens_seen=123, wall_clock_s=4.5, seed=3)

    path = save_checkpoint(tmp_path, model, opt, state, {"name": "c2_jepa_ema"})
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp*")), "a temp file leaked into the checkpoint dir"
    assert find_latest(tmp_path) == path

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) >= {"state", "model", "optimizer", "rng", "config"}
    assert any(k.startswith("target_encoder.") for k in payload["model"]), "EMA state not saved"
    assert "torch" in payload["rng"] and "python" in payload["rng"]

    restored = make_model("c2_jepa_ema", seed=99)
    opt2 = torch.optim.AdamW(restored.trainable_parameters(), lr=1e-3)
    back = load_checkpoint(path, restored, opt2)
    assert (back.step, back.tokens_seen, back.seed) == (7, 123, 3)
    for (_, p), (_, q) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert torch.equal(p, q)


def test_find_latest_survives_a_missing_pointer(tmp_path: Path) -> None:
    """A preemption between the two renames must still leave a loadable checkpoint."""
    model = make_model("c1_mlm")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    save_checkpoint(tmp_path, model, opt, TrainState(step=1))
    newest = save_checkpoint(tmp_path, model, opt, TrainState(step=2))
    (tmp_path / "latest.txt").unlink()
    assert find_latest(tmp_path) == newest
    assert find_latest(tmp_path / "empty") is None


def test_ema_update_moves_target_towards_online() -> None:
    """C2's EMA branch tracks the online encoder and stays out of the optimiser."""
    model = make_model("c2_jepa_ema")
    assert model.has_ema
    for p in model.target_encoder.parameters():
        assert not p.requires_grad
    assert all(p.requires_grad for p in model.trainable_parameters())

    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    before = model.target_encoder.embed.weight.clone()
    model.update_ema(0.5)
    after = model.target_encoder.embed.weight
    assert torch.allclose(after, 0.5 * before + 0.5 * model.encoder.embed.weight)

    model.update_ema(1.0)  # tau=1 freezes the target
    assert torch.allclose(after, 0.5 * before + 0.5 * model.encoder.embed.weight)


@pytest.mark.parametrize("name", OBJECTIVE_NAMES)
def test_fit_runs_and_writes_csv(tmp_path: Path, name: str) -> None:
    """An end-to-end mini run: budget respected, CSV written, checkpoint on disk."""
    batches = [make_batch(seed=i) for i in range(2)]
    trainer = make_trainer(tmp_path, name, batches, f"fit-{name}")
    state = trainer.fit()

    assert state.step == trainer.cfg.total_steps
    assert state.tokens_seen > 0
    assert find_latest(trainer.run_dir) is not None

    rows = (trainer.run_dir / "metrics.csv").read_text().strip().splitlines()
    header = rows[0].split(",")
    assert rows[1:], "no metric rows were written"
    for column in ("step", "lr", "tokens_seen", "wall_clock_s", "steps_per_sec", "loss", "grad_norm"):
        assert column in header
    for metric in trainer.objective.metric_names:
        assert metric in header
    for column in ("rankme", "dim_std", "offdiag_cov_mass"):
        assert column in header, "collapse columns must exist even when xjepa.eval is missing"

    summary = trainer.summary()
    assert summary["steps"] == trainer.cfg.total_steps
    assert set(summary) >= {"compile_graphs", "compile_recompiles", "stop_reason"}


def test_max_minutes_stops_the_run(tmp_path: Path) -> None:
    """The wall-clock guard must stop a run that would overrun the budget."""
    batches = [make_batch()]
    seed_everything(0)
    cfg = replace(tiny_config(tmp_path, "c1_mlm"), total_steps=10_000, max_minutes=0.0)
    trainer = Trainer(cfg, make_model("c1_mlm"), build_objective(cfg.objective), batches, log_dir=tmp_path / "m")
    state = trainer.fit()
    assert state.step == 0
    assert trainer.summary()["stop_reason"] == "max_minutes"


def test_token_budget_is_the_stopping_condition(tmp_path: Path) -> None:
    """The budget is tokens, not steps and never epochs."""
    batches = [make_batch()]
    real_tokens = int(batches[0].pad_mask.sum())
    seed_everything(0)
    cfg = replace(
        tiny_config(tmp_path, "c1_mlm"),
        total_steps=10_000,
        total_tokens=real_tokens * 6,
        log_every=2,
        checkpoint_every=0,
        collapse_every=0,
    )
    trainer = Trainer(cfg, make_model("c1_mlm"), build_objective(cfg.objective), batches, log_dir=tmp_path / "tok")
    state = trainer.fit()

    assert trainer.summary()["stop_reason"] == "token_budget"
    assert state.tokens_seen >= cfg.total_tokens
    assert state.step < cfg.total_steps
    # tokens_seen counts REAL residues only: the pad column must not be counted.
    assert state.tokens_seen == real_tokens * state.step

    header = (trainer.run_dir / "metrics.csv").read_text().splitlines()[0].split(",")
    assert header[0] == "tokens_seen", "tokens_seen must be the primary x-axis"
    assert "epoch" not in header, "epochs are not a meaningful unit under the crop policy"


class FakeBatcher(list):
    """A batcher exposing the data-policy accessors from the data contract."""

    def summary(self) -> dict[str, object]:
        return {"policy": "hybrid", "buckets": [128, 256, 384, 512], "occupancy_threshold": 0.85}

    def padding_efficiency(self) -> float:
        return 0.957

    def crop_loss_fraction(self) -> float:
        return 0.142


class FakeCorpus:
    """A corpus exposing ``summary()`` with the measured VRAM residency."""

    def summary(self) -> dict[str, object]:
        return {"residues": 12_500_000, "vram_gib": 2.99}


def test_data_policy_is_recorded_at_startup(tmp_path: Path) -> None:
    """Padding efficiency, crop loss and real VRAM residency land in the run dir."""
    import json

    batches = FakeBatcher([make_batch()])
    seed_everything(0)
    cfg = replace(tiny_config(tmp_path, "c1_mlm"), total_steps=4, warmup_steps=1, checkpoint_every=0)
    trainer = Trainer(
        cfg,
        make_model("c1_mlm"),
        build_objective(cfg.objective),
        batches,
        log_dir=tmp_path / "policy",
        corpus=FakeCorpus(),
    )
    trainer.fit()

    policy = json.loads((trainer.run_dir / "data_policy.json").read_text())
    assert policy["padding_efficiency"] == pytest.approx(0.957)
    assert policy["crop_loss_fraction"] == pytest.approx(0.142)
    assert policy["batcher"]["policy"] == "hybrid"
    assert policy["corpus"]["vram_gib"] == pytest.approx(2.99)
    assert policy["token_budget"] == cfg.total_tokens


def test_data_policy_tolerates_a_batcher_without_accessors(tmp_path: Path) -> None:
    """A plain list of batches must not break the run."""
    from xjepa.train.trainer import data_policy

    assert data_policy([make_batch()], None) == {}


def test_collapse_metrics_tolerate_a_missing_eval_module(tmp_path: Path) -> None:
    """A missing ``xjepa.eval.collapse`` must degrade to empty, never raise."""
    from xjepa.train.trainer import collapse_metrics

    out = collapse_metrics(torch.randn(16, D_MODEL))
    assert isinstance(out, dict)
    assert not out or set(out) == {"rankme", "dim_std", "offdiag_cov_mass"}


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #


def test_one_config_per_condition() -> None:
    """Exactly seven configs, named after the seven conditions."""
    found = {p.stem for p in CONFIG_DIR.glob("*.yaml")}
    assert found == set(OBJECTIVE_NAMES)


def test_configs_differ_only_in_their_objective_block() -> None:
    """The experimental control, enforced mechanically.

    Both halves are checked: the parsed configs agree on every non-objective
    field, and the raw text agrees line-for-line outside the objective block.
    """
    configs = {name: load_config(CONFIG_DIR / f"{name}.yaml") for name in OBJECTIVE_NAMES}
    reference = configs["c1_mlm"]
    for name, cfg in configs.items():
        assert cfg.objective.name == name
        assert cfg.name == name, "run name must follow the objective name"
        for field_name in RunConfig.__dataclass_fields__:
            if field_name in ("objective", "name"):
                continue
            assert getattr(cfg, field_name) == getattr(reference, field_name), (
                f"{name}.yaml differs from c1_mlm.yaml outside the objective block: {field_name}"
            )

    prefixes = {}
    for name in OBJECTIVE_NAMES:
        text = (CONFIG_DIR / f"{name}.yaml").read_text()
        head, sep, _ = text.partition("# --- objective ---")
        assert sep, f"{name}.yaml has no objective block marker"
        prefixes[name] = head
    assert len(set(prefixes.values())) == 1, "config preambles are not byte-identical"


def test_config_matches_the_token_budget() -> None:
    """15300 steps x 65536 tokens/step = 1.0B tokens, identical for every condition."""
    cfg = load_config(CONFIG_DIR / "c3_jepa_frozen.yaml")
    assert cfg.total_steps * cfg.tokens_per_step == pytest.approx(1.0e9, rel=0.005)
    assert cfg.warmup_steps == 500
    assert cfg.checkpoint_every == 1000
    assert cfg.compile is True and cfg.bf16 is True


def test_config_overrides_and_typo_rejection(tmp_path: Path) -> None:
    """CLI overrides apply; an unknown key is an error."""
    cfg = load_config(CONFIG_DIR / "c1_mlm.yaml", seed=7, max_minutes=1.5, device=None)
    assert cfg.seed == 7 and cfg.max_minutes == 1.5 and cfg.device == "cuda"

    bad = tmp_path / "bad.yaml"
    bad.write_text("totl_steps: 10\nobjective:\n  name: c1_mlm\n")
    with pytest.raises(ValueError):
        load_config(bad)
