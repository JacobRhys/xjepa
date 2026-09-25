"""Pure-PyTorch stand-ins for the two ``torch_scatter`` functions ESM-IF1 needs.

``fair-esm``'s inverse-folding code has exactly one hard dependency on the
compiled ``torch_scatter`` extension::

    esm/inverse_folding/gvp_modules.py:34:
        from torch_scatter import scatter_add, scatter

``torch_scatter`` ships as a C++/CUDA extension whose wheels are built against
one exact torch+CUDA pair. On a recent image (torch 2.8 + cu128) a prebuilt
wheel often does not exist, and building from source needs nvcc and 20+ minutes
of a billed GPU instance -- to obtain two functions that PyTorch has had native
equivalents for since 1.12.

So: implement them, register the module, move on. :func:`install` puts this
module into ``sys.modules`` under the name ``torch_scatter`` *before* ESM-IF1 is
imported, but only when the real package is absent -- if a genuine
``torch_scatter`` is installed it always wins.

The implementations follow ``torch_scatter``'s documented broadcasting
semantics. ``scatter_add`` is exact (``Tensor.scatter_add_``); ``mean`` is
computed as sum/count, and ``max``/``min`` use ``scatter_reduce_`` with
``include_self=False`` so empty output slots stay at the identity rather than
being polluted by the initial fill.
"""

from __future__ import annotations

import sys
from typing import Optional

import torch
from torch import Tensor

__all__ = ["broadcast", "scatter_add", "scatter", "install", "is_real_package_available"]


def broadcast(src: Tensor, other: Tensor, dim: int) -> Tensor:
    """Expand an index tensor to ``other``'s shape, per torch_scatter semantics.

    Args:
        src: The index tensor, usually 1-D.
        other: The tensor being scattered, whose shape ``src`` must match.
        dim: Scatter dimension (may be negative).

    Returns:
        ``src`` expanded to ``other.size()``.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand_as(other)


def _output(
    src: Tensor, index: Tensor, dim: int, out: Optional[Tensor], dim_size: Optional[int], fill: float
) -> Tensor:
    if out is not None:
        return out
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return torch.full(size, fill, dtype=src.dtype, device=src.device)


def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    """Sum ``src`` into positions given by ``index`` along ``dim``."""
    index = broadcast(index, src, dim)
    target = _output(src, index, dim if dim >= 0 else src.dim() + dim, out, dim_size, 0.0)
    return target.scatter_add_(dim, index, src)


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    """Scatter ``src`` into ``index`` along ``dim`` with the given reduction.

    Supports ``"sum"``/``"add"``, ``"mean"``, ``"max"``/``"amax"`` and
    ``"min"``/``"amin"`` -- the reductions ``torch_scatter`` exposes and the only
    ones ESM-IF1's GVP layers use.
    """
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim, out, dim_size)

    idx = broadcast(index, src, dim)
    pos_dim = dim if dim >= 0 else src.dim() + dim

    if reduce == "mean":
        total = scatter_add(src, index, dim, None, dim_size)
        ones = torch.ones_like(src)
        count = scatter_add(ones, index, dim, None, total.size(pos_dim))
        result = total / count.clamp_min(1)
        if out is not None:
            return out.copy_(result)
        return result

    if reduce in ("max", "amax"):
        op, fill = "amax", float("-inf")
    elif reduce in ("min", "amin"):
        op, fill = "amin", float("inf")
    else:
        raise ValueError(f"unsupported reduce {reduce!r}")

    target = _output(src, idx, pos_dim, out, dim_size, fill)
    target = target.scatter_reduce_(dim, idx, src, reduce=op, include_self=False)
    # torch_scatter returns 0 where nothing was scattered, not +/-inf.
    return torch.nan_to_num(target, posinf=0.0, neginf=0.0)


def is_real_package_available() -> bool:
    """True when a genuine compiled ``torch_scatter`` can be imported."""
    if "torch_scatter" in sys.modules:
        return getattr(sys.modules["torch_scatter"], "__xjepa_shim__", False) is False
    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        return False
    return True


def install(force: bool = False) -> str:
    """Register this module as ``torch_scatter`` when the real one is missing.

    Must be called *before* importing ``esm.inverse_folding``.

    Args:
        force: Shim even if the real package is importable. For testing only.

    Returns:
        ``"real"`` if the genuine package is in use, ``"shim"`` if this module
        was registered.
    """
    if not force and is_real_package_available():
        return "real"
    module = sys.modules[__name__]
    module.__xjepa_shim__ = True  # type: ignore[attr-defined]
    sys.modules["torch_scatter"] = module
    return "shim"
