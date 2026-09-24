"""Sync-free training loop for the seven-condition study.

Design rules (``docs/CONTRACTS.md``) and how they are met here:

**1. Zero device syncs in the hot loop.** :meth:`Trainer.train_step` never calls
``.item()``, ``.cpu()``, ``.numpy()``, ``float()`` or ``print()`` on a tensor.
Every metric is added into :class:`MetricBuffer`, a single preallocated
``[n_metrics]`` tensor living on the training device. The host reads it once
every ``log_every`` steps, so the CPU stays ahead of the GPU and the launch queue
never drains. ``tests/test_train.py::test_no_sync_calls_in_step`` scans the
source of the step function and fails the build if a sync call reappears.

**2. Fixed shapes.** The batcher yields four bucket shapes (128/256/384/512), so
``torch.compile(..., dynamic=False)`` caches a small, bounded set of graphs --
see :func:`compile_stats` for the recompile count reported at the end of a run.

**3. Precision.** bf16 autocast for the forward/backward; parameters and
optimiser state stay fp32 (master weights). No GradScaler: bf16 has fp32's
exponent range, so loss scaling is unnecessary.

**4. The budget is tokens.** 1.0B tokens, identical for every condition. Never
epochs: the batcher random-crops part of each sequence per pass, so an epoch is
not a fixed quantity of data and would mean different things under different
bucket policies. ``tokens_seen`` is the stopping condition and the primary
x-axis of ``metrics.csv``.

**5. Budget guards.** ``--max-minutes`` is a hard wall-clock stop and checkpoints
are written every ``checkpoint_every`` steps atomically, so a spot preemption
costs at most one checkpoint interval (RESEARCH_PLAN.md sec. 6.1).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch import Tensor, nn

from xjepa.train.checkpoint import TrainState, find_latest, load_checkpoint, save_checkpoint
from xjepa.train.objectives import (
    OBJECTIVE_NAMES,
    Objective,
    ObjectiveConfig,
    build_objective,
)
from xjepa.train.schedule import LrSchedule, ScheduleConfig, ema_tau_at

__all__ = [
    "RunConfig",
    "apply_overrides",
    "MetricBuffer",
    "XJepaModel",
    "Trainer",
    "build_model",
    "build_optimizer",
    "compile_stats",
    "load_config",
    "seed_everything",
    "main",
]


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    """One run of the study.

    Every field except :attr:`objective` is identical across the seven condition
    configs -- that identity *is* the experimental control, so the YAML files are
    structured to differ only in their ``objective:`` block.
    """

    # --- identity -------------------------------------------------------- #
    name: str = "c1_mlm"
    seed: int = 0
    out_dir: str = "runs"

    # --- objective (THE ONLY BLOCK THAT DIFFERS BETWEEN CONDITIONS) ------- #
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    # --- data ------------------------------------------------------------ #
    corpus_path: str = "data/corpus"
    target_dim: int = 128
    tie_mlm_weights: bool = True
    tokens_per_step: int = 65_536
    buckets: tuple[int, ...] = (128, 256, 384, 512)
    mask_rate: float = 0.15
    #: "random" (15% per residue, ESM-2) or "span" (geometric, mean 8) -- the
    #: masking ablation of RESEARCH_PLAN.md sec. 4, run on C3 only.
    mask_mode: str = "random"
    mean_span: float = 8.0
    #: Bucket policy: "hybrid" (pad up at >=0.85 occupancy, else crop),
    #: "crop" (zero padding, reaches the flash SDPA backend), or "pad".
    #: Settled by scripts/pilot.py; must be identical across all conditions.
    bucket_policy: str = "hybrid"

    # --- model ----------------------------------------------------------- #
    n_layers: int = 6
    d_model: int = 320
    n_heads: int = 20
    d_ff: int = 1280
    vocab: int = 33
    max_len: int = 512
    rope: bool = True

    # --- optimisation ---------------------------------------------------- #
    # The budget is TOKENS. `total_steps` is the schedule horizon and a
    # belt-and-braces cap; whichever of the two binds first ends the run. Epochs
    # are deliberately absent: the batcher's `hybrid` policy random-crops part of
    # each sequence per pass, so an epoch is not a fixed quantity of data.
    total_tokens: int = 1_000_000_000
    total_steps: int = 15_300
    base_lr: float = 4e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.0
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    grad_clip: float = 1.0

    # --- runtime --------------------------------------------------------- #
    device: str = "cuda"
    compile: bool = True
    bf16: bool = True
    max_minutes: float = 35.0

    # --- logging / checkpointing ----------------------------------------- #
    log_every: int = 50
    collapse_every: int = 500
    checkpoint_every: int = 1000

    @property
    def schedule(self) -> ScheduleConfig:
        """The LR schedule implied by this config."""
        return ScheduleConfig(
            base_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-friendly view of the config (stored in checkpoints)."""
        d = {k: v for k, v in self.__dict__.items() if k != "objective"}
        d["objective"] = dict(self.objective.__dict__)
        return d


def load_config(path: str | os.PathLike[str], **overrides: Any) -> RunConfig:
    """Load a run config from YAML.

    The YAML uses the same nesting as :class:`RunConfig`: a flat body plus an
    ``objective:`` mapping. Unknown keys raise, so a typo in a config can never
    silently change nothing.

    Args:
        path: Path to a ``configs/*.yaml`` file.
        **overrides: Fields to override (e.g. ``seed=1``); ``None`` values ignored.

    Returns:
        The parsed :class:`RunConfig`.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    obj_raw = dict(raw.pop("objective", {}) or {})
    valid = set(RunConfig.__dataclass_fields__)
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

    if "buckets" in raw and raw["buckets"] is not None:
        raw["buckets"] = tuple(int(b) for b in raw["buckets"])
    if "betas" in raw and raw["betas"] is not None:
        raw["betas"] = tuple(float(b) for b in raw["betas"])

    obj_valid = set(ObjectiveConfig.__dataclass_fields__)
    obj_unknown = set(obj_raw) - obj_valid
    if obj_unknown:
        raise ValueError(f"unknown objective keys in {path}: {sorted(obj_unknown)}")

    objective = ObjectiveConfig(**obj_raw)
    # The run name defaults to the objective name so that the seven condition
    # configs need no top-level `name:` and stay diff-identical outside their
    # objective block.
    raw.setdefault("name", objective.name)
    cfg = RunConfig(objective=objective, **raw)
    clean = {k: v for k, v in overrides.items() if v is not None}
    if clean:
        cfg = replace(cfg, **clean)
    return cfg


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG stream for a reproducible run.

    Args:
        seed: Base seed (the ``--seed`` flag).
        deterministic: Also disable cuDNN autotuning nondeterminism. Left on by
            default; it costs little at this model size and the study reports
            seed variance as a first-class result (RESEARCH_PLAN.md sec. 1.7).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:  # pragma: no cover - numpy optional
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# model assembly
# --------------------------------------------------------------------------- #


class XJepaModel(nn.Module):
    """Encoder plus exactly the heads the chosen objective needs.

    Sub-modules an objective does not use are ``None``, never merely unused: an
    unused module in the optimiser would add optimiser state, perturb the
    parameter count between conditions, and make "did the predictor receive a
    gradient?" untestable. C5 / C5b therefore genuinely have no predictor.

    Attributes:
        encoder: The sequence encoder.
        predictor: Narrow JEPA predictor (C2/C3/C4/C5c) or ``None``.
        mlm_head: Vocabulary head (C1/C4) or ``None``.
        latent_head: Linear ``d_model -> target_dim`` (C2 target side, C5, C5b) or ``None``.
        target_encoder: EMA copy of ``encoder`` (C2) or ``None``.
        target_head: EMA copy of ``latent_head`` (C2) or ``None``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module | None = None,
        mlm_head: nn.Module | None = None,
        latent_head: nn.Module | None = None,
        target_encoder: nn.Module | None = None,
        target_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.mlm_head = mlm_head
        self.latent_head = latent_head
        self.target_encoder = target_encoder
        self.target_head = target_head
        if target_encoder is not None:
            for p in target_encoder.parameters():
                p.requires_grad_(False)
        if target_head is not None:
            for p in target_head.parameters():
                p.requires_grad_(False)

    @property
    def has_ema(self) -> bool:
        """Whether an EMA target branch exists (C2 only)."""
        return self.target_encoder is not None

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that require grad, i.e. everything but the EMA branch."""
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        """In-place EMA update of the target branch: ``t <- tau*t + (1-tau)*o``.

        Implemented with ``torch._foreach_*`` so the whole branch updates in two
        fused multi-tensor kernels. ``tau`` is a Python float from
        :func:`xjepa.train.schedule.ema_tau_at` -- no tensor is read, so this
        costs no device sync.

        Args:
            tau: Momentum in ``[0, 1]``; 1.0 freezes the target.
        """
        pairs: list[tuple[list[Tensor], list[Tensor]]] = []
        if self.target_encoder is not None:
            pairs.append((list(self.target_encoder.parameters()), list(self.encoder.parameters())))
        if self.target_head is not None and self.latent_head is not None:
            pairs.append((list(self.target_head.parameters()), list(self.latent_head.parameters())))
        for tgt, src in pairs:
            if not tgt:
                continue
            torch._foreach_mul_(tgt, tau)
            torch._foreach_add_(tgt, src, alpha=1.0 - tau)
        # Buffers (e.g. RoPE caches, running stats) are copied, not averaged.
        if self.target_encoder is not None:
            for tb, sb in zip(self.target_encoder.buffers(), self.encoder.buffers()):
                tb.copy_(sb)


def build_model(cfg: RunConfig, objective: Objective) -> XJepaModel:
    """Build the encoder and only the heads ``objective`` needs.

    ``xjepa.model`` is imported lazily so that this module (and the trainer
    tests) import cleanly while another agent is still writing that package.

    Args:
        cfg: Run config supplying the encoder geometry.
        objective: The objective, whose ``needs_*`` flags select the heads.

    Returns:
        The assembled :class:`XJepaModel` (on CPU; the caller moves it).
    """
    import copy

    from xjepa.model.encoder import Encoder, EncoderConfig
    from xjepa.model.heads import MlmHead, Predictor, PredictorConfig

    enc_cfg = EncoderConfig(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        vocab=cfg.vocab,
        max_len=cfg.max_len,
        rope=cfg.rope,
    )
    encoder = Encoder(enc_cfg)

    # The heads take their own config objects, not EncoderConfig. Derive the
    # predictor geometry from the encoder so the two stay consistent when the
    # encoder is rescaled, and keep the narrow defaults otherwise (I-JEPA).
    pred_cfg = PredictorConfig(
        d_in=enc_cfg.d_model,
        d_out=cfg.target_dim,
        max_len=enc_cfg.max_len,
        rope=enc_cfg.rope,
    )
    predictor = (
        Predictor(pred_cfg, replace_masked=objective.predictor_replaces_masked)
        if objective.needs_predictor
        else None
    )
    # tie_weights=True requires the embedding matrix to tie against.
    mlm_head = (
        MlmHead(
            d_model=enc_cfg.d_model,
            vocab=enc_cfg.vocab,
            tie_weights=cfg.tie_mlm_weights,
            embed_weight=encoder.embed_tokens.weight if cfg.tie_mlm_weights else None,
        )
        if objective.needs_mlm_head
        else None
    )
    latent_head = nn.Linear(cfg.d_model, cfg.target_dim) if objective.needs_latent_head else None

    target_encoder = None
    target_head = None
    if objective.needs_ema:
        # C2 only. The contract fixes the predictor's output at `target_dim`
        # while the encoder emits `d_model`, so the target branch needs a
        # projection. It is a *frozen random* projection (rank-preserving, and
        # it adds no trainable parameters that could differ between
        # conditions); the target encoder itself is what the EMA tracks.
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


def build_optimizer(model: XJepaModel, cfg: RunConfig) -> torch.optim.AdamW:
    """AdamW with multi-tensor kernels and no weight decay on 1-D parameters.

    .. note:: **Contract conflict.** ``docs/CONTRACTS.md`` rule 4 asks for
       ``foreach=True, fused=True`` together, but ``torch.optim`` rejects that
       combination outright ("`fused` and `foreach` cannot be `True` together").
       ``fused`` already implies a single multi-tensor kernel and is the faster
       of the two, so this builds ``fused=True`` and falls back to
       ``foreach=True`` only where fused is unavailable.

    Args:
        model: Model whose trainable parameters to optimise.
        cfg: Run config supplying the AdamW hyper-parameters.

    Returns:
        The configured optimiser.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for p in model.trainable_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, Any] = {
        "lr": cfg.base_lr,
        "betas": tuple(cfg.betas),
        "eps": cfg.eps,
    }
    try:
        return torch.optim.AdamW(groups, fused=True, **kwargs)
    except (RuntimeError, ValueError):
        return torch.optim.AdamW(groups, foreach=True, **kwargs)


class ModelView:
    """The sub-module handles an objective calls, optionally ``torch.compile``d.

    ``torch.compile(composite_module)`` would be a silent no-op for this codebase:
    objectives call ``model.encoder(...)`` / ``model.predictor(...)`` directly, and
    attribute access on an ``OptimizedModule`` forwards to the *uncompiled*
    ``_orig_mod`` children. Compiling each leaf instead keeps the state dict keys
    clean (the compiled wrappers are held here, not assigned back onto the
    ``nn.Module``) while still tracing the whole forward.

    Attributes mirror :class:`~xjepa.train.objectives.TrainableModel`.
    """

    def __init__(self, model: XJepaModel, compile_modules: bool) -> None:
        def wrap(module: nn.Module | None) -> nn.Module | None:
            if module is None or not compile_modules:
                return module
            return torch.compile(module, dynamic=False)

        self.encoder = wrap(model.encoder)
        self.predictor = wrap(model.predictor)
        self.mlm_head = wrap(model.mlm_head)
        self.latent_head = wrap(model.latent_head)
        self.target_encoder = wrap(model.target_encoder)
        self.target_head = wrap(model.target_head)


def compile_stats() -> dict[str, int]:
    """Summarise ``torch._dynamo`` counters for the end-of-run report.

    Expected values for this study: one graph per (bucket, sub-module) pair with
    static shapes, i.e. 4 buckets x {encoder, head(s)} -- roughly 8-12 graphs and
    **zero** recompiles after the first pass over the bucket set. A growing
    ``recompiles`` count means a shape leaked into the hot loop.

    Returns:
        ``{"graphs": ..., "recompiles": ..., "graph_breaks": ...}``.
    """
    try:
        from torch._dynamo.utils import counters
    except Exception:  # pragma: no cover - dynamo always present in torch 2.x
        return {"graphs": 0, "recompiles": 0, "graph_breaks": 0}

    def total(bucket: str) -> int:
        return int(sum(counters.get(bucket, {}).values()))

    frames = counters.get("frames", {})
    return {
        "graphs": int(frames.get("ok", 0)),
        "recompiles": total("recompile_reasons") or total("recompiles"),
        "graph_breaks": total("graph_break"),
    }


# --------------------------------------------------------------------------- #
# metric accumulation
# --------------------------------------------------------------------------- #


class MetricBuffer:
    """Preallocated ``[n_metrics]`` device tensor of running sums.

    ``add`` performs in-place index adds; nothing crosses the PCIe bus. ``flush``
    is the *only* place a value is moved to the host, and the trainer calls it
    once every ``log_every`` steps.
    """

    def __init__(self, names: Sequence[str], device: torch.device) -> None:
        self.names = tuple(names)
        self._index = {n: i for i, n in enumerate(self.names)}
        self._buf = torch.zeros(len(self.names), dtype=torch.float32, device=device)
        self.count = 0

    def __len__(self) -> int:
        return len(self.names)

    @property
    def buffer(self) -> Tensor:
        """The raw accumulator (for tests / introspection)."""
        return self._buf

    def add(self, name: str, value: Tensor) -> None:
        """Add one scalar tensor into the accumulator slot ``name``."""
        self._buf[self._index[name]] += value.detach().to(torch.float32).reshape(())

    def add_many(self, values: dict[str, Tensor]) -> None:
        """Add a metrics dict; unknown names are ignored (fixed-size buffer)."""
        for name, value in values.items():
            idx = self._index.get(name)
            if idx is not None:
                self._buf[idx] += value.detach().to(torch.float32).reshape(())

    def tick(self) -> None:
        """Record that one step contributed to the accumulator (host-side int)."""
        self.count += 1

    def flush(self) -> dict[str, float]:
        """Sync once, return per-step means, and zero the accumulator.

        Returns:
            ``{name: mean_over_accumulated_steps}`` as Python floats.
        """
        n = max(self.count, 1)
        values = (self._buf / n).tolist()  # the single host transfer
        self._buf.zero_()
        self.count = 0
        return dict(zip(self.names, values))


class CsvLogger:
    """Append-only CSV logger -- no wandb, no network, no paid tooling.

    The header is fixed at construction (union of trainer fields, objective
    metric names and the collapse diagnostics) so that every row has the same
    columns and the file loads with a one-line ``pandas.read_csv``.
    """

    def __init__(self, path: str | os.PathLike[str], columns: Sequence[str]) -> None:
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns, extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        """Write one row and flush (rows are rare; durability beats buffering)."""
        self._writer.writerow({k: row.get(k, "") for k in self.columns})
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._fh.closed:
            self._fh.close()


def collapse_metrics(h: Tensor) -> dict[str, float]:
    """Collapse diagnostics for an encoder-output sample, or ``{}`` if unavailable.

    ``xjepa.eval.collapse`` is imported lazily and defensively: a missing eval
    module must never take down a training run that is burning paid GPU time.

    Args:
        h: float ``[N, d]`` sample of encoder outputs (already on any device).

    Returns:
        ``{"rankme": ..., "dim_std": ..., "offdiag_cov_mass": ...}`` or ``{}``.
    """
    try:
        from xjepa.eval.collapse import dim_std, offdiag_cov_mass, rankme
    except Exception:
        return {}
    try:
        x = h.detach().float()
        return {
            "rankme": float(rankme(x)),
            "dim_std": float(dim_std(x)),
            "offdiag_cov_mass": float(offdiag_cov_mass(x)),
        }
    except Exception:
        return {}


COLLAPSE_COLUMNS = ("rankme", "dim_std", "offdiag_cov_mass")


def data_policy(batches: Any, corpus: Any | None = None) -> dict[str, Any]:
    """Snapshot the data policy a run actually executed under.

    The bucket policy is not a neutral detail: the default ``hybrid`` policy pads
    a sequence up when bucket occupancy is >= 0.85 and random-crops it down
    otherwise (measured ~95.7% padding efficiency, ~14.2% of residues cropped per
    pass). Because the crop window is redrawn on device each pass, full coverage
    happens *across* passes, not within one -- which is precisely why the budget
    is counted in tokens and never in epochs.

    Equally, ``GpuCorpus.summary()`` reports the *measured* VRAM residency
    (~2.99 GiB at 12.5M residues, ~3.6 GiB at 15M), which the "< 3 GB" table in
    ``docs/CONTRACTS.md`` understates. Every run records the real figure.

    Every probe is optional and defensive: a missing accessor must not take down
    a run that is burning paid GPU time.

    Args:
        batches: The batcher (usually ``BucketBatcher``).
        corpus: The ``GpuCorpus``, if the caller has a handle on it.

    Returns:
        A JSON-serialisable dict; missing fields are simply absent.
    """
    policy: dict[str, Any] = {}

    def probe(obj: Any, attr: str, key: str) -> None:
        fn = getattr(obj, attr, None)
        if fn is None:
            return
        try:
            value = fn() if callable(fn) else fn
        except Exception:
            return
        try:
            policy[key] = value if isinstance(value, (int, float, str, bool, dict, list)) else repr(value)
        except Exception:  # pragma: no cover - exotic repr
            pass

    probe(batches, "summary", "batcher")
    probe(batches, "padding_efficiency", "padding_efficiency")
    probe(batches, "crop_loss_fraction", "crop_loss_fraction")
    if corpus is not None:
        probe(corpus, "summary", "corpus")
    return policy


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #


class Trainer:
    """Drives one condition of the study to its fixed token budget.

    Args:
        cfg: The run config.
        model: The assembled model (already on the target device).
        objective: The loss for this condition.
        batches: Any iterable of ``Batch`` objects resident on device -- in
            production ``xjepa.data.store.BucketBatcher``, in tests a list. It is
            iterated directly: the batcher's interleave order is a static list of
            Python ints, so nothing here may wrap it in something that reads a
            device tensor per step.
        log_dir: Where ``metrics.csv`` and checkpoints are written. Defaults to
            ``cfg.out_dir/cfg.name-seed<N>``.
        corpus: Optional ``GpuCorpus``, used only to record its ``summary()``
            (real VRAM residency etc.) in ``data_policy.json`` at startup.
    """

    def __init__(
        self,
        cfg: RunConfig,
        model: XJepaModel,
        objective: Objective,
        batches: Iterable[Any],
        log_dir: str | os.PathLike[str] | None = None,
        corpus: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.objective = objective
        self.batches = batches
        self.corpus = corpus
        self.device = torch.device(cfg.device)
        self.run_dir = Path(log_dir) if log_dir is not None else Path(cfg.out_dir) / f"{cfg.name}-seed{cfg.seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = build_optimizer(model, cfg)
        self.schedule = LrSchedule(cfg.schedule)
        self.view = ModelView(model, compile_modules=cfg.compile and self.device.type == "cuda")
        self.state = TrainState(seed=cfg.seed)

        self.metric_names: tuple[str, ...] = ("loss", "grad_norm", "tokens") + tuple(objective.metric_names)
        self.metrics = MetricBuffer(self.metric_names, self.device)

        self._autocast_enabled = cfg.bf16 and self.device.type == "cuda"
        self._last_hidden: Tensor | None = None
        self._stop_reason = "completed"

        self.logger = CsvLogger(
            self.run_dir / "metrics.csv",
            columns=[
                "tokens_seen",  # primary x-axis: the budget is tokens, not epochs
                "step",
                "lr",
                "wall_clock_s",
                "steps_per_sec",
                *self.metric_names,
                *COLLAPSE_COLUMNS,
            ],
        )

    # ------------------------------------------------------------------ #
    # the hot loop
    # ------------------------------------------------------------------ #

    def train_step(self, batch: Any, lr: float, tau: float) -> None:
        """One optimiser step. **Contains no host<->device synchronisation.**

        Everything that could sync -- reading the loss, the grad norm, the token
        count -- is instead accumulated into :attr:`metrics`, a device-resident
        tensor. ``lr`` and ``tau`` arrive as Python floats computed from the
        integer step counter, so setting them reads no tensor either.

        Args:
            batch: A device-resident ``Batch``.
            lr: Learning rate for this step (from the schedule).
            tau: EMA momentum for this step (ignored unless the model has an EMA).
        """
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self._autocast_enabled,
        ):
            loss, metrics = self.objective.loss(self.view, batch)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.trainable_parameters(), self.cfg.grad_clip, foreach=True
        )
        self.optimizer.step()

        if self.model.has_ema:
            self.model.update_ema(tau)

        self.metrics.add("loss", loss)
        self.metrics.add("grad_norm", grad_norm)
        self.metrics.add("tokens", batch.pad_mask.sum())
        self.metrics.add_many(metrics)
        self.metrics.tick()

    # ------------------------------------------------------------------ #
    # orchestration
    # ------------------------------------------------------------------ #

    def fit(self) -> TrainState:
        """Run until the token budget, the step budget or ``--max-minutes`` hits.

        Returns:
            The final :class:`TrainState`.
        """
        self.model.train()
        deadline = time.monotonic() + self.cfg.max_minutes * 60.0
        start = time.monotonic()
        window_start = start
        window_steps = 0
        resume_offset = self.state.wall_clock_s

        self._write_data_policy()

        for batch in self._batch_stream():
            # TOKENS are the budget (1.0B), not epochs: the `hybrid` bucket
            # policy random-crops ~14% of residues per pass, so an "epoch" means
            # different things under different bucket policies and is never a
            # stopping condition or an axis here. `tokens_seen` advances at the
            # logging cadence, which is the only place the accumulator is read.
            if self.cfg.total_tokens > 0 and self.state.tokens_seen >= self.cfg.total_tokens:
                self._stop_reason = "token_budget"
                break
            if self.state.step >= self.cfg.total_steps:
                self._stop_reason = "step_budget"
                break
            if time.monotonic() >= deadline:
                self._stop_reason = "max_minutes"
                break

            step = self.state.step
            lr = self.schedule.value(step)
            tau = ema_tau_at(
                step,
                self.cfg.total_steps,
                self.objective.cfg.ema_tau_base,
                self.objective.cfg.ema_tau_final,
            )
            self.train_step(batch, lr, tau)
            self._last_batch = batch

            self.state.step = step + 1
            window_steps += 1

            if self.cfg.log_every > 0 and self.state.step % self.cfg.log_every == 0:
                now = time.monotonic()
                steps_per_sec = window_steps / max(now - window_start, 1e-9)
                self.state.wall_clock_s = resume_offset + (now - start)
                self._log(lr, steps_per_sec)
                window_start, window_steps = now, 0

            if self.cfg.checkpoint_every > 0 and self.state.step % self.cfg.checkpoint_every == 0:
                self.state.wall_clock_s = resume_offset + (time.monotonic() - start)
                self.save()
        else:
            self._stop_reason = "data_exhausted"

        self.state.wall_clock_s = resume_offset + (time.monotonic() - start)
        if self.metrics.count:
            self._log(self.schedule.value(max(self.state.step - 1, 0)), 0.0)
        self.save()
        self.logger.close()
        return self.state

    def _write_data_policy(self) -> Path | None:
        """Record the data policy this run executed under, once, at startup.

        Written as ``data_policy.json`` next to ``metrics.csv`` so that padding
        efficiency, crop loss and the measured corpus VRAM residency are part of
        every run's artefacts rather than a number in a doc.

        Returns:
            The path written, or ``None`` if nothing could be probed.
        """
        import json

        policy = data_policy(self.batches, self.corpus)
        if not policy:
            return None
        policy["token_budget"] = self.cfg.total_tokens
        policy["tokens_per_step_nominal"] = self.cfg.tokens_per_step
        policy["buckets"] = list(self.cfg.buckets)
        path = self.run_dir / "data_policy.json"
        path.write_text(json.dumps(policy, indent=2, default=repr), encoding="utf-8")
        return path

    def _batch_stream(self) -> Iterator[Any]:
        """Yield batches, restarting the iterable when it is exhausted.

        ``BucketBatcher`` is an infinite iterator in production; a finite list in
        tests is cycled so short unit runs still reach their step budget.
        """
        while True:
            empty = True
            for batch in self.batches:
                empty = False
                yield batch
            if empty:
                return

    def _log(self, lr: float, steps_per_sec: float) -> None:
        """Flush the metric buffer (one sync) and append a CSV row."""
        window_steps = max(self.metrics.count, 1)
        means = self.metrics.flush()
        tokens_per_step = means.pop("tokens", 0.0)
        # `tokens` accumulates as a per-step mean; convert back to a window total.
        self.state.tokens_seen += int(round(tokens_per_step * window_steps))
        row: dict[str, Any] = {
            "step": self.state.step,
            "lr": lr,
            "tokens_seen": self.state.tokens_seen,
            "wall_clock_s": round(self.state.wall_clock_s, 3),
            "steps_per_sec": round(steps_per_sec, 4),
            "tokens": tokens_per_step,
            **means,
        }
        if self.cfg.collapse_every > 0 and self.state.step % self.cfg.collapse_every == 0:
            row.update(self._collapse_probe())
        self.logger.log(row)

    @torch.no_grad()
    def _collapse_probe(self) -> dict[str, float]:
        """Compute collapse diagnostics on the most recent batch's encoder output.

        Runs the *uncompiled* encoder so this diagnostic can never add a graph to
        the compiled cache, and never runs inside :meth:`train_step`.
        """
        batch = getattr(self, "_last_batch", None)
        if batch is None:
            return {}
        self.model.eval()
        try:
            h = self.model.encoder(batch.tokens, batch.pad_mask)
            flat = h.reshape(-1, h.shape[-1])[batch.pad_mask.reshape(-1)]
            return collapse_metrics(flat)
        except Exception:
            return {}
        finally:
            self.model.train()

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #

    def save(self) -> Path:
        """Write an atomic checkpoint of the full training state."""
        return save_checkpoint(self.run_dir, self.model, self.optimizer, self.state, self.cfg.to_dict())

    def resume(self, path: str | os.PathLike[str] | None = None) -> bool:
        """Restore from ``path`` (or the newest checkpoint in the run dir).

        Args:
            path: Explicit checkpoint path, or ``None`` to auto-discover.

        Returns:
            True if a checkpoint was loaded.
        """
        ckpt = Path(path) if path is not None else find_latest(self.run_dir)
        if ckpt is None or not Path(ckpt).is_file():
            return False
        self.state = load_checkpoint(ckpt, self.model, self.optimizer, map_location=self.device)
        return True

    def summary(self) -> dict[str, Any]:
        """End-of-run report: budget consumed, stop reason and compile counters."""
        stats = compile_stats()
        return {
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            "steps": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "token_budget": self.cfg.total_tokens,
            "wall_clock_s": round(self.state.wall_clock_s, 2),
            "stop_reason": self._stop_reason,
            "compile_graphs": stats["graphs"],
            "compile_recompiles": stats["recompiles"],
            "compile_graph_breaks": stats["graph_breaks"],
        }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_batches(cfg: RunConfig, objective: Objective | None = None):
    """Construct the on-device corpus, masker and bucket batcher (lazy import).

    Per the contract there is no ``DataLoader``: the corpus lives in VRAM and
    batching is index arithmetic on device.

    The corruption scheme follows the objective rather than the config, because
    it is not a free parameter: MLM conditions need ESM-2's 80/10/10, latent
    conditions need every masked position replaced by ``<mask>``, and the
    unmasked controls (C5, C5c) need no masker at all -- with ``masker=None``
    the batcher emits all-false ``mask_sel`` and all ``-100`` labels, so
    ``original_tokens`` recovers the clean sequence.

    Args:
        cfg: Run config.
        objective: The built objective, used to pick the corruption scheme and
            whether to mask at all. ``None`` falls back to plain JEPA masking.

    Returns:
        ``(corpus, batcher)`` -- the corpus is returned so the trainer can record
        its ``summary()`` (measured VRAM residency) rather than trusting the
        table in ``docs/CONTRACTS.md``.
    """
    from xjepa.data.masking import MaskingConfig, Masker
    from xjepa.data.store import BucketBatcher, GpuCorpus

    corpus = GpuCorpus.load(cfg.corpus_path, device=cfg.device, target_dim=cfg.target_dim)

    masker = None
    if objective is None or objective.uses_masking:
        corruption = "mlm" if (objective is not None and objective.needs_mlm_head) else "jepa"
        masker = Masker(
            MaskingConfig(
                mode=cfg.mask_mode,
                rate=cfg.mask_rate,
                mean_span=cfg.mean_span,
                corruption=corruption,
            ),
            generator=torch.Generator(device=cfg.device).manual_seed(cfg.seed),
        )

    batcher = BucketBatcher(
        corpus,
        buckets=cfg.buckets,
        token_budget=cfg.tokens_per_step,
        masker=masker,
        seed=cfg.seed,
        policy=cfg.bucket_policy,
    )
    return corpus, batcher


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    p = argparse.ArgumentParser(description="Train one condition of the X-JEPA protein study.")
    p.add_argument("--config", required=True, help="Path to a configs/*.yaml file.")
    p.add_argument("--seed", type=int, default=None, help="Base seed; overrides the config.")
    p.add_argument("--max-minutes", type=float, default=None, help="Hard wall-clock stop (budget guard).")
    p.add_argument("--out-dir", type=str, default=None, help="Root directory for runs.")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu")
    p.add_argument("--total-steps", type=int, default=None, help="Override the step budget (pilot runs).")
    p.add_argument("--total-tokens", type=int, default=None, help="Override the token budget (pilot runs).")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile (debugging).")
    p.add_argument("--resume", action="store_true", help="Resume from the newest checkpoint in the run dir.")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
        help=(
            "Override one config field, repeatable. Dotted keys reach into the "
            "objective block (e.g. --set objective.lambda_jepa=0.1). Values are "
            "parsed as YAML, so numbers and booleans keep their types. Used by "
            "scripts/run_grid.py for the lambda sweep and the target variants."
        ),
    )
    return p.parse_args(argv)


def apply_overrides(cfg: RunConfig, overrides: Sequence[str]) -> RunConfig:
    """Apply ``--set KEY=VALUE`` overrides to a loaded config.

    Unknown keys raise rather than silently doing nothing: a typo in a grid
    override would otherwise produce a run that looks like the variant you asked
    for and is actually the baseline.

    Args:
        cfg: The loaded run config.
        overrides: ``"key=value"`` strings; ``objective.`` prefixes the objective block.

    Returns:
        A new :class:`RunConfig`.
    """
    import yaml

    top: dict[str, Any] = {}
    obj: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        if key.startswith("objective."):
            field_name = key.split(".", 1)[1]
            if field_name not in ObjectiveConfig.__dataclass_fields__:
                raise SystemExit(f"--set: unknown objective field {field_name!r}")
            obj[field_name] = value
        else:
            if key not in RunConfig.__dataclass_fields__:
                raise SystemExit(f"--set: unknown config field {key!r}")
            top[key] = value

    if obj:
        top["objective"] = replace(cfg.objective, **obj)
    return replace(cfg, **top) if top else cfg


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: build everything, train, print a one-line summary.

    Args:
        argv: Argument vector, or ``None`` for ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    cfg = load_config(
        args.config,
        seed=args.seed,
        max_minutes=args.max_minutes,
        out_dir=args.out_dir,
        device=args.device,
        total_steps=args.total_steps,
        total_tokens=args.total_tokens,
    )
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    if args.no_compile:
        cfg = replace(cfg, compile=False)
    if cfg.objective.name not in OBJECTIVE_NAMES:
        raise SystemExit(f"config objective {cfg.objective.name!r} is not one of {OBJECTIVE_NAMES}")

    seed_everything(cfg.seed)
    objective = build_objective(cfg.objective)
    model = build_model(cfg, objective).to(cfg.device)
    corpus, batcher = build_batches(cfg, objective)
    trainer = Trainer(cfg, model, objective, batcher, corpus=corpus)
    if args.resume:
        trainer.resume()

    trainer.fit()
    summary = trainer.summary()
    # Persisted as the run's completion marker: scripts/run_grid.py treats the
    # presence of summary.json as "this cell finished", which is what makes the
    # grid resumable after a spot preemption.
    with open(trainer.run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({**summary, "config": cfg.to_dict()}, fh, indent=2, default=str)
    print(" ".join(f"{k}={v}" for k, v in summary.items()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
