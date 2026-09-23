"""ESM-2 t6 (\"8M\") style transformer encoder, trained from random init.

Architecture decisions and why
------------------------------
* **LayerNorm, not RMSNorm.** The reference configuration we are replicating is
  ESM-2 t6, which uses LayerNorm. This is a *controlled* study of training
  objectives, so every architectural degree of freedom is pinned to the
  reference; swapping in RMSNorm would save ~1% of step time and confound the
  comparison with a (small) architecture change. Cost: 2 * d_model extra
  parameters per norm and one extra mean reduction.
* **GELU FFN, not SwiGLU.** Same reasoning: ESM-2 uses a plain
  ``Linear -> GELU -> Linear`` FFN with ``d_ff = 4 * d_model``. SwiGLU would
  change the parameter budget (3 matrices instead of 2) and the effective width.
* **Pre-norm blocks** with a final norm after the stack: strictly better
  optimisation behaviour without warmup tuning at this depth, and what
  ESM-2 does (``emb_layer_norm_after``).
* **RoPE instead of learned absolute positions.** Per the study config; also
  removes ``max_len * d_model`` parameters and makes the model length-agnostic
  across the 128/256/384/512 buckets.

Performance rules honoured here
-------------------------------
* Attention is **only** ``F.scaled_dot_product_attention``. The padding mask is
  converted to an **additive float mask in the query dtype** so the fused
  mem-efficient kernel is eligible; a bool or dtype-mismatched mask silently
  drops SDPA to the math backend.
* No ``.item()``, no ``.cpu()``, no data-dependent control flow, no Python-side
  shape branching -> ``torch.compile`` produces one graph per bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import RotaryEmbedding

__all__ = [
    "EncoderConfig",
    "Encoder",
    "EncoderBlock",
    "MultiHeadSelfAttention",
    "FeedForward",
    "build_additive_mask",
    "activation_dtype",
]


def activation_dtype(x: Tensor) -> torch.dtype:
    """Dtype the fused matmuls will actually run in.

    Under ``torch.autocast`` the first ``nn.Linear`` demotes activations to
    bf16, but the embedding output (and therefore a mask built from it) is still
    fp32. Building the attention bias in *this* dtype means SDPA never has to
    reject a dtype-mismatched mask.
    """
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


@dataclass
class EncoderConfig:
    """Encoder hyper-parameters. Defaults are the ESM-2 t6 reference config."""

    n_layers: int = 6
    d_model: int = 320
    n_heads: int = 20
    d_ff: int = 1280
    vocab: int = 33
    max_len: int = 512
    rope: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )
        if self.rope and (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")

    @property
    def head_dim(self) -> int:
        """Channels per attention head."""
        return self.d_model // self.n_heads


def build_additive_mask(pad_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Turn a boolean key-padding mask into an additive attention bias.

    Args:
        pad_mask: ``[B, L]`` bool, ``True`` = real residue (per the data contract).
        dtype: dtype of the queries that this mask will be added to. **Must**
            match, otherwise ``scaled_dot_product_attention`` refuses the fused
            kernels and falls back to the math implementation.

    Returns:
        ``[B, 1, 1, L]`` float bias, ``0`` on real positions and
        ``finfo(dtype).min`` on padding.

    Note:
        ``finfo.min`` rather than ``-inf``: a fully padded row would produce
        ``nan`` after the softmax with ``-inf`` (and ``nan`` poisons the whole
        backward), while ``finfo.min`` degrades gracefully to a uniform
        attention over the padding, whose output is discarded anyway.
    """
    neg = torch.finfo(dtype).min
    bias = torch.zeros(pad_mask.shape, dtype=dtype, device=pad_mask.device)
    bias = bias.masked_fill(~pad_mask, neg)
    return bias[:, None, None, :]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE, routed through fused SDPA.

    Args:
        d_model: model width.
        n_heads: number of attention heads.
        rope: shared :class:`~xjepa.model.rope.RotaryEmbedding`, or ``None`` to
            run without positional rotation.
    """

    def __init__(self, d_model: int, n_heads: int, rope: RotaryEmbedding | None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Fused qkv: one GEMM instead of three, identical parameter count.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.rope = rope

    def forward(self, x: Tensor, attn_bias: Tensor | None) -> Tensor:
        """Run attention.

        Args:
            x: ``[B, L, d_model]`` input activations.
            attn_bias: ``[B, 1, 1, L]`` additive mask in ``x``'s dtype, or ``None``.

        Returns:
            ``[B, L, d_model]``.
        """
        b, l, _ = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.n_heads, self.head_dim)
        # [3, B, H, L, Dh]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            q, k = self.rope(q, k)

        if attn_bias is not None and attn_bias.dtype != q.dtype:
            # Guard rail: a dtype mismatch here is the classic silent
            # fallback-to-math-kernel bug.
            attn_bias = attn_bias.to(q.dtype)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        o = o.transpose(1, 2).reshape(b, l, self.n_heads * self.head_dim)
        return self.out(o)


class FeedForward(nn.Module):
    """Position-wise GELU FFN (ESM-2 layout).

    Args:
        d_model: model width.
        d_ff: hidden width.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the FFN to ``[..., d_model]`` and return the same shape."""
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    """Pre-norm transformer block: ``x + attn(ln(x))`` then ``x + ffn(ln(x))``.

    Args:
        d_model: model width.
        n_heads: attention heads.
        d_ff: FFN hidden width.
        rope: shared rotary embedding (or ``None``).
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, rope: RotaryEmbedding | None
    ) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rope)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor, attn_bias: Tensor | None) -> Tensor:
        """Run one block over ``[B, L, d_model]``."""
        x = x + self.attn(self.ln_attn(x), attn_bias)
        x = x + self.ffn(self.ln_ffn(x))
        return x


class Encoder(nn.Module):
    """The protein sequence encoder.

    Args:
        cfg: :class:`EncoderConfig`.
        device: device the module (including the RoPE tables) is built on.
        dtype: parameter dtype. Keep fp32 master weights and use bf16 autocast
            for the forward pass.
    """

    def __init__(
        self,
        cfg: EncoderConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cfg = cfg = cfg or EncoderConfig()
        factory = {"device": device, "dtype": dtype}

        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.d_model, **factory)
        self.rope: RotaryEmbedding | None = (
            RotaryEmbedding(cfg.head_dim, cfg.max_len, device=device)
            if cfg.rope
            else None
        )
        self.layers = nn.ModuleList(
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, self.rope)
            for _ in range(cfg.n_layers)
        )
        self.ln_final = nn.LayerNorm(cfg.d_model, **factory)
        self.to(device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ESM-2 style init: normal(0, 0.02) on weights, zeros on biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def param_count(self, trainable_only: bool = False) -> int:
        """Total number of parameters (buffers such as RoPE tables excluded)."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def forward(self, tokens: Tensor, pad_mask: Tensor) -> Tensor:
        """Encode a fixed-shape batch.

        Args:
            tokens: ``[B, L]`` int64 token ids.
            pad_mask: ``[B, L]`` bool, ``True`` = real residue.

        Returns:
            ``[B, L, d_model]`` contextual representations. Padded positions are
            zeroed so that downstream reductions cannot pick up junk; real
            positions are provably independent of anything at padded positions.
        """
        x = self.embed_tokens(tokens)
        attn_bias = build_additive_mask(pad_mask, activation_dtype(x))
        for layer in self.layers:
            x = layer(x, attn_bias)
        x = self.ln_final(x)
        return x * pad_mask.unsqueeze(-1).to(x.dtype)
