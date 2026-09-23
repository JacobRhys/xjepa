"""Learning-rate and EMA-momentum schedules.

Every schedule here is **pure Python arithmetic on the integer step counter**.
Nothing in this module reads a tensor, so setting the learning rate costs zero
device syncs (assigning a Python float into ``param_group["lr"]`` is a host-side
dict write, not a device transfer).

See ``docs/CONTRACTS.md`` -- hard rule 1: no ``.item()`` / ``.cpu()`` in the
training step. A tensor-valued LR schedule would violate it, so the schedules
are deliberately scalar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

__all__ = ["ScheduleConfig", "lr_at", "ema_tau_at", "LrSchedule"]


@dataclass(frozen=True)
class ScheduleConfig:
    """Hyper-parameters for the linear-warmup / cosine-decay LR schedule.

    Attributes:
        base_lr: Peak learning rate reached at the end of warmup.
        warmup_steps: Number of steps of linear warmup from ``0`` to ``base_lr``.
        total_steps: Total number of optimiser steps in the run.
        min_lr_ratio: Floor of the cosine decay, as a fraction of ``base_lr``.
    """

    base_lr: float = 4e-4
    warmup_steps: int = 500
    total_steps: int = 15_300
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be < total_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must lie in [0, 1]")


def lr_at(step: int, cfg: ScheduleConfig) -> float:
    """Return the learning rate for a 0-based optimiser ``step``.

    Linear warmup for ``cfg.warmup_steps`` steps, then cosine decay to
    ``cfg.base_lr * cfg.min_lr_ratio`` at ``cfg.total_steps``.

    Args:
        step: 0-based step index. Values beyond ``total_steps`` clamp to the floor.
        cfg: Schedule configuration.

    Returns:
        The learning rate as a Python float.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    floor = cfg.base_lr * cfg.min_lr_ratio

    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        # step 0 gets a non-zero LR so the very first update is not a no-op.
        return cfg.base_lr * (step + 1) / cfg.warmup_steps

    decay_steps = cfg.total_steps - cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / decay_steps
    if progress >= 1.0:
        return floor
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (cfg.base_lr - floor) * cosine


def ema_tau_at(step: int, total_steps: int, tau_base: float = 0.996, tau_final: float = 1.0) -> float:
    """EMA momentum for the C2 target encoder, cosine-annealed ``tau_base -> tau_final``.

    This follows the BYOL / I-JEPA convention (RESEARCH_PLAN.md sec. 4, row 2:
    "tau 0.996 -> 1.0"): the target encoder is frozen progressively as training
    proceeds.

    Args:
        step: 0-based optimiser step.
        total_steps: Total steps in the run.
        tau_base: Momentum at step 0.
        tau_final: Momentum at the final step.

    Returns:
        The momentum as a Python float in ``[tau_base, tau_final]``.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(step / total_steps, 0.0), 1.0)
    return tau_final - (tau_final - tau_base) * 0.5 * (1.0 + math.cos(math.pi * progress))


class LrSchedule:
    """Stateless LR schedule bound to an optimiser's param groups.

    The schedule holds no tensors and no optimiser state, so it needs no entry in
    the checkpoint: the LR is a pure function of ``step``, which *is* checkpointed.
    """

    def __init__(self, cfg: ScheduleConfig) -> None:
        self.cfg = cfg

    def value(self, step: int) -> float:
        """Learning rate at ``step``."""
        return lr_at(step, self.cfg)

    def apply(self, param_groups: Iterable[dict], step: int) -> float:
        """Write the LR for ``step`` into every param group.

        Args:
            param_groups: The optimiser's ``param_groups`` list.
            step: 0-based optimiser step.

        Returns:
            The learning rate that was applied (a Python float, for CSV logging).
        """
        lr = lr_at(step, self.cfg)
        for group in param_groups:
            group["lr"] = lr
        return lr
