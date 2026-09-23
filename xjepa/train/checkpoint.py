"""Crash-safe checkpointing.

Runs live on preemptible community-cloud instances (RESEARCH_PLAN.md sec. 6.1:
"checkpoint every 1,000 steps so preemption costs <= 1.5 min"), so a checkpoint
must never be observable in a half-written state:

1. serialise into ``<name>.tmp.<pid>`` **in the destination directory** (same
   filesystem, so the rename is atomic);
2. ``flush`` + ``os.fsync`` the file;
3. ``os.replace`` onto the final path -- atomic on POSIX;
4. rewrite the ``latest`` pointer the same way, and fsync the directory.

A preemption at any point leaves either the previous checkpoint or the new one,
never a truncated file.

Everything needed for an **exact** resume is saved: step, token count, model
(including the EMA target encoder, which is a registered sub-module), optimiser
state, and the Python / torch / CUDA / numpy RNG states.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = ["TrainState", "rng_state", "restore_rng_state", "save_checkpoint", "load_checkpoint", "find_latest"]

LATEST_POINTER = "latest.txt"


@dataclass
class TrainState:
    """Scalar training progress that must survive a preemption.

    Attributes:
        step: Number of *completed* optimiser steps.
        tokens_seen: Number of real (non-pad) tokens consumed so far.
        wall_clock_s: Accumulated training wall-clock across all resumes.
        seed: The run's base seed, for provenance.
    """

    step: int = 0
    tokens_seen: int = 0
    wall_clock_s: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for serialisation."""
        return {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "wall_clock_s": self.wall_clock_s,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainState":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            step=int(d["step"]),
            tokens_seen=int(d["tokens_seen"]),
            wall_clock_s=float(d["wall_clock_s"]),
            seed=int(d.get("seed", 0)),
        )


def rng_state() -> dict[str, Any]:
    """Capture every RNG stream the training loop can consume.

    Returns:
        A picklable dict with the Python, torch-CPU, torch-CUDA and (if present)
        numpy generator states.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:  # pragma: no cover - numpy is optional
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the RNG streams captured by :func:`rng_state`.

    Missing streams are skipped (e.g. resuming a CUDA run on a CPU box), so a
    checkpoint stays loadable across machines.

    Args:
        state: A dict produced by :func:`rng_state`.
    """
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].to(torch.uint8).cpu())
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:  # pragma: no cover - differing device count
            pass
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:  # pragma: no cover - numpy is optional
            pass


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Serialise ``payload`` to ``path`` atomically (temp file + fsync + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # only reachable if torch.save or replace raised
            tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)


def _atomic_write_text(text: str, path: Path) -> None:
    """Write ``text`` to ``path`` atomically."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename survives a power loss (best effort)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - not supported on every filesystem
        pass


def save_checkpoint(
    directory: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write a resumable checkpoint atomically and update the ``latest`` pointer.

    Args:
        directory: Checkpoint directory; created if absent.
        model: The model to serialise. If it has been wrapped by
            ``torch.compile``, the *original* module's state dict is stored (via
            ``_orig_mod``) so the file loads with or without compilation.
        optimizer: Optimiser whose state to serialise.
        state: Scalar training progress.
        config: Optional run config, stored for provenance.

    Returns:
        Path to the written checkpoint file.
    """
    directory = Path(directory)
    target = getattr(model, "_orig_mod", model)
    payload = {
        "format": 1,
        "state": state.to_dict(),
        "model": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_state(),
        "config": config or {},
    }
    path = directory / f"ckpt_{state.step:08d}.pt"
    _atomic_torch_save(payload, path)
    _atomic_write_text(path.name, directory / LATEST_POINTER)
    return path


def load_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> TrainState:
    """Restore a checkpoint in place and return the training state.

    Args:
        path: Checkpoint file written by :func:`save_checkpoint`.
        model: Model to load into (compiled wrappers are unwrapped).
        optimizer: Optimiser to load into, or ``None`` to skip.
        map_location: Device to map storages onto.
        restore_rng: Whether to restore the RNG streams (True for an exact resume).

    Returns:
        The :class:`TrainState` stored in the checkpoint.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng and "rng" in payload:
        restore_rng_state(payload["rng"])
    return TrainState.from_dict(payload["state"])


def find_latest(directory: str | os.PathLike[str]) -> Path | None:
    """Return the newest checkpoint in ``directory``, or ``None`` if there is none.

    Prefers the ``latest.txt`` pointer; falls back to the highest-numbered
    ``ckpt_*.pt`` if the pointer is missing or dangling (e.g. preempted between
    the checkpoint rename and the pointer rename).

    Args:
        directory: Checkpoint directory.

    Returns:
        Path to the newest usable checkpoint, or ``None``.
    """
    directory = Path(directory)
    pointer = directory / LATEST_POINTER
    if pointer.is_file():
        candidate = directory / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_file():
            return candidate
    candidates = sorted(directory.glob("ckpt_*.pt"))
    return candidates[-1] if candidates else None
