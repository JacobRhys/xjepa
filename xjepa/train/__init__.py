"""Training package: objectives, schedules, checkpointing and the trainer.

Public surface (see ``docs/CONTRACTS.md``):

* :class:`~xjepa.train.objectives.Objective` and :func:`~xjepa.train.objectives.build_objective`
  -- the seven conditions of the study.
* :class:`~xjepa.train.trainer.Trainer` -- the sync-free training loop.
* :mod:`xjepa.train.schedule` -- host-side LR / EMA schedules.
* :mod:`xjepa.train.checkpoint` -- atomic, preemption-safe checkpoints.

Only :mod:`xjepa.train.objectives`, :mod:`xjepa.train.schedule` and
:mod:`xjepa.train.checkpoint` are re-exported eagerly; the trainer is imported
lazily so that importing this package never pulls in ``xjepa.data`` /
``xjepa.model``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xjepa.train.checkpoint import TrainState, find_latest, load_checkpoint, save_checkpoint
from xjepa.train.objectives import (
    OBJECTIVE_NAMES,
    Objective,
    ObjectiveConfig,
    build_objective,
)
from xjepa.train.schedule import LrSchedule, ScheduleConfig, ema_tau_at, lr_at

if TYPE_CHECKING:  # pragma: no cover
    from xjepa.train.trainer import MetricBuffer, RunConfig, Trainer, XJepaModel

__all__ = [
    "OBJECTIVE_NAMES",
    "Objective",
    "ObjectiveConfig",
    "build_objective",
    "LrSchedule",
    "ScheduleConfig",
    "ema_tau_at",
    "lr_at",
    "TrainState",
    "find_latest",
    "load_checkpoint",
    "save_checkpoint",
    "RunConfig",
    "Trainer",
    "XJepaModel",
    "MetricBuffer",
]

_LAZY = {"RunConfig", "Trainer", "XJepaModel", "MetricBuffer"}


def __getattr__(name: str) -> Any:
    """Import trainer symbols on first access (keeps package import cheap)."""
    if name in _LAZY:
        from xjepa.train import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
