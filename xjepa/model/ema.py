"""Exponential-moving-average target encoder (condition ``c2_jepa_ema``).

The target encoder is a frozen copy of the online encoder whose weights track it
with momentum ``m``:

``theta_target <- m * theta_target + (1 - m) * theta_online``

with ``m`` annealed ``0.996 -> 1.0`` on a cosine schedule over training, the
BYOL/I-JEPA recipe: fast tracking early (the target is garbage anyway), a frozen
target late (so the student chases a stationary objective).

Efficiency
----------
The update is two ``torch._foreach_*`` calls over the whole parameter list, not
a Python loop over ~80 tensors. On a model this small the per-tensor kernel
launch overhead would otherwise dominate the update. Everything runs under
``torch.no_grad()``; the momentum is computed from a host-side ``int`` step
counter, so the update issues **no** device->host sync.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

__all__ = ["cosine_momentum", "EmaTargetEncoder"]


def cosine_momentum(
    step: int,
    total_steps: int,
    base: float = 0.996,
    final: float = 1.0,
) -> float:
    """Cosine momentum schedule from ``base`` at step 0 to ``final`` at the end.

    Args:
        step: current optimiser step (0-based).
        total_steps: number of steps the schedule spans. Must be > 0.
        base: momentum at step 0.
        final: momentum at (and after) ``total_steps``.

    Returns:
        The momentum as a plain Python float -- no tensor, hence no sync.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    t = min(max(step, 0), total_steps) / total_steps
    return final - (final - base) * (1.0 + math.cos(math.pi * t)) / 2.0


class EmaTargetEncoder(nn.Module):
    """Frozen EMA copy of an online module.

    Args:
        online: the module to track. It is deep-copied once; the copy's
            parameters are detached and ``requires_grad_(False)``.
        total_steps: horizon of the momentum schedule.
        base_momentum: momentum at step 0.
        final_momentum: momentum at the end of the schedule.
    """

    def __init__(
        self,
        online: nn.Module,
        total_steps: int,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
    ) -> None:
        super().__init__()
        self.total_steps = total_steps
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum

        self.target: nn.Module = copy.deepcopy(online)
        self.target.requires_grad_(False)
        self.target.eval()

        # Parameter lists are captured ONCE, in a matching order, so the update
        # is a pure foreach over two static lists.
        self._online_params: list[Tensor] = list(online.parameters())
        self._target_params: list[Tensor] = list(self.target.parameters())
        if len(self._online_params) != len(self._target_params):
            raise RuntimeError("online/target parameter lists diverged")
        for po, pt in zip(self._online_params, self._target_params):
            if po.shape != pt.shape:
                raise RuntimeError("online/target parameter shapes diverged")

        # Buffers (e.g. RoPE tables) are deterministic functions of the config,
        # so they are copied once and never averaged.
        self._online_buffers: list[Tensor] = list(online.buffers())
        self._target_buffers: list[Tensor] = list(self.target.buffers())

    @property
    def module(self) -> nn.Module:
        """The wrapped target module."""
        return self.target

    def momentum(self, step: int) -> float:
        """Momentum for ``step`` under this instance's schedule."""
        return cosine_momentum(
            step, self.total_steps, self.base_momentum, self.final_momentum
        )

    @torch.no_grad()
    def update(self, step: int) -> float:
        """Apply one EMA step.

        Args:
            step: current optimiser step (0-based), a host-side ``int``.

        Returns:
            The momentum that was applied (a Python float, for logging).
        """
        m = self.momentum(step)
        torch._foreach_mul_(self._target_params, m)
        torch._foreach_add_(self._target_params, self._online_params, alpha=1.0 - m)
        return m

    @torch.no_grad()
    def copy_from_online(self) -> None:
        """Hard-reset the target to the online weights (and buffers)."""
        torch._foreach_copy_(self._target_params, self._online_params)
        if self._target_buffers:
            for bt, bo in zip(self._target_buffers, self._online_buffers):
                bt.copy_(bo)

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        """Run the target module under ``no_grad`` (it is never trained).

        Returns:
            Whatever the wrapped module returns -- for an
            :class:`~xjepa.model.encoder.Encoder`, ``[B, L, d_model]``.
        """
        with torch.no_grad():
            return self.target(*args, **kwargs)  # type: ignore[no-any-return]
