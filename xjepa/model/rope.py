"""Rotary positional embeddings (RoPE).

Design notes
------------
* The cos/sin tables for ``max_len`` positions are built **once** in ``__init__``
  on the requested device/dtype and stored as *non-persistent* buffers, so they
  travel with ``.to(device)`` but never enter a checkpoint. ``forward`` only
  slices them -- no trig is evaluated inside the training step.
* Tables are kept in fp32 and cast to the query dtype at application time. The
  cast is a cheap elementwise op on a ``[1, 1, L, Dh]`` tensor, and it keeps the
  angle table exact under bf16 autocast (bf16 has 8 mantissa bits; storing
  ``cos(theta)`` in bf16 costs ~2e-3 absolute error at every position).
* Convention is the GPT-NeoX / ESM-2 "rotate-half" layout: channel ``i`` is
  paired with channel ``i + Dh/2``. This matches ``esm.rotary_embedding``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["RotaryEmbedding", "rotate_half", "apply_rope"]


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dimension: ``[a, b] -> [-b, a]``.

    Args:
        x: tensor whose last dimension is even.

    Returns:
        Tensor of the same shape as ``x``.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply a precomputed rotation to ``x``.

    Args:
        x: ``[B, H, L, Dh]`` queries or keys.
        cos: ``[1, 1, L, Dh]`` cosine table, already in ``x``'s dtype.
        sin: ``[1, 1, L, Dh]`` sine table, already in ``x``'s dtype.

    Returns:
        Rotated tensor of shape ``[B, H, L, Dh]``.
    """
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed rotary embedding tables for a fixed maximum length.

    Args:
        head_dim: per-head channel count. Must be even.
        max_len: maximum sequence length that will ever be requested.
        base: geometric base of the inverse-frequency schedule (10000 in ESM-2).
        device: device the tables are built on. Build them where they are used;
            building on CPU and relying on a later ``.to()`` wastes a copy and,
            worse, is easy to forget for non-persistent buffers.
        dtype: dtype of the stored tables (fp32 recommended, see module docstring).
    """

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(
        self,
        head_dim: int,
        max_len: int,
        base: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
        )  # [Dh/2]
        pos = torch.arange(max_len, device=device, dtype=torch.float32)  # [L]
        freqs = torch.outer(pos, inv_freq)  # [L, Dh/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, Dh]

        # Non-persistent: derived deterministically from (head_dim, max_len, base),
        # so there is no reason to ship them in a state_dict.
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate queries and keys in place-free fashion.

        Args:
            q: ``[B, H, L, Dh]`` queries.
            k: ``[B, H, L, Dh]`` keys (same ``L`` as ``q``).

        Returns:
            ``(q_rot, k_rot)``, both ``[B, H, L, Dh]`` and in ``q``/``k``'s dtype.
        """
        seq_len = q.shape[-2]
        # Pure slicing -- no host sync, no recomputation, torch.compile friendly
        # because ``seq_len`` is static per bucket.
        cos = self.cos_cached[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"head_dim={self.head_dim}, max_len={self.max_len}, base={self.base}"
