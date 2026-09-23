"""Prediction heads: the narrow JEPA predictor and the MLM head.

Both heads are deliberately thin. In I-JEPA the predictor is the *only* place
where the "guess the target representation" work happens, and keeping it narrow
(d=160 against the encoder's 320) is what stops the encoder from offloading its
representation quality into the predictor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder import EncoderBlock, build_additive_mask, activation_dtype
from .rope import RotaryEmbedding

__all__ = ["PredictorConfig", "Predictor", "MlmHead"]


@dataclass
class PredictorConfig:
    """Predictor hyper-parameters (defaults per the research contract)."""

    d_in: int = 320
    n_layers: int = 2
    d_model: int = 160
    n_heads: int = 4
    d_ff: int = 640
    d_out: int = 128
    max_len: int = 512
    rope: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.rope and (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")

    @property
    def head_dim(self) -> int:
        """Channels per predictor attention head."""
        return self.d_model // self.n_heads


class Predictor(nn.Module):
    """Narrow transformer that maps encoder states to target-space predictions.

    The predictor attends over the **whole** sequence (context positions carry
    the information) but the loss is only ever read at masked positions -- the
    objective does that selection, so this module returns dense ``[B, L, d_out]``
    output and never gathers, which keeps shapes static for ``torch.compile``.

    Masked positions are overwritten with a learned "mask query" embedding
    before the predictor blocks run, so the predictor cannot copy the encoder's
    own (possibly leaky) state at the position it is asked to predict.
    Positional identity is restored inside the blocks by RoPE.

    Args:
        cfg: :class:`PredictorConfig`.
        replace_masked: if ``False`` the mask query is never applied and the
            predictor simply transforms every encoder state (condition
            ``c5c_predictor_nomask``).
        device: build device.
        dtype: parameter dtype.
    """

    def __init__(
        self,
        cfg: PredictorConfig | None = None,
        replace_masked: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cfg = cfg = cfg or PredictorConfig()
        self.replace_masked = replace_masked
        factory = {"device": device, "dtype": dtype}

        self.in_proj = nn.Linear(cfg.d_in, cfg.d_model, bias=True, **factory)
        # Learned query used at positions we are asked to predict.
        self.mask_query = nn.Parameter(torch.zeros(cfg.d_model, **factory))
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
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_out, bias=True, **factory)
        self.to(device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Small-normal init, matching the encoder."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.mask_query, mean=0.0, std=0.02)

    def param_count(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        h: Tensor,
        mask_sel: Tensor | None = None,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict target-space vectors for every position.

        Args:
            h: ``[B, L, d_in]`` encoder output.
            mask_sel: ``[B, L]`` bool, ``True`` = masked position. ``None``
                disables the mask query entirely.
            pad_mask: ``[B, L]`` bool, ``True`` = real residue. Optional but
                strongly recommended: without it the predictor attends to
                padding. (The contract signature omits it; it is keyword
                optional here so both call styles work.)

        Returns:
            ``[B, L, d_out]`` predictions. The objective reads the masked rows.
        """
        x = self.in_proj(h)
        if self.replace_masked and mask_sel is not None:
            # torch.where keeps shapes static -- no boolean indexing, no sync.
            x = torch.where(
                mask_sel.unsqueeze(-1),
                self.mask_query.to(x.dtype).expand_as(x),
                x,
            )
        attn_bias = (
            None
            if pad_mask is None
            else build_additive_mask(pad_mask, activation_dtype(x))
        )
        for layer in self.layers:
            x = layer(x, attn_bias)
        return self.out_proj(self.ln_final(x))


class MlmHead(nn.Module):
    """Masked-language-model head (ESM-2 / RoBERTa layout).

    ``Linear(d, d) -> GELU -> LayerNorm -> Linear(d, vocab) + bias``.

    Args:
        d_model: encoder width.
        vocab: vocabulary size.
        tie_weights: if ``True`` the output projection *shares* the token
            embedding matrix (``embed_weight`` must then be supplied). Tying is
            the ESM-2 default and removes ``vocab * d_model`` parameters; it
            also couples input and output representations, which matters for a
            comparison against non-MLM objectives -- hence the explicit flag.
        embed_weight: the ``[vocab, d_model]`` embedding weight to tie to.
        device: build device.
        dtype: parameter dtype.
    """

    def __init__(
        self,
        d_model: int = 320,
        vocab: int = 33,
        tie_weights: bool = True,
        embed_weight: Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if tie_weights and embed_weight is None:
            raise ValueError("tie_weights=True requires embed_weight")
        if embed_weight is not None and tuple(embed_weight.shape) != (vocab, d_model):
            raise ValueError(
                f"embed_weight shape {tuple(embed_weight.shape)} != {(vocab, d_model)}"
            )
        factory = {"device": device, "dtype": dtype}
        self.tie_weights = tie_weights
        self.dense = nn.Linear(d_model, d_model, bias=True, **factory)
        self.ln = nn.LayerNorm(d_model, **factory)
        self.bias = nn.Parameter(torch.zeros(vocab, **factory))
        if tie_weights:
            assert embed_weight is not None
            # Hidden in a list so nn.Module does NOT re-register the embedding
            # weight as a parameter of this head: it stays owned by the
            # embedding, and the shared storage means gradients accumulate into
            # the one tensor exactly as ESM-2 intends.
            self._tied_ref: list[Tensor] = [embed_weight]
            self._untied = None
        else:
            self._tied_ref = []
            self._untied = nn.Parameter(torch.empty(vocab, d_model, **factory))
        self.reset_parameters()

    @property
    def decoder_weight(self) -> Tensor:
        """The output projection matrix (tied to the embedding, or private)."""
        if self.tie_weights:
            return self._tied_ref[0]
        assert self._untied is not None
        return self._untied

    def reset_parameters(self) -> None:
        """Init the head. The tied weight is owned by the embedding, not reset."""
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)
        nn.init.zeros_(self.bias)
        if self._untied is not None:
            nn.init.normal_(self._untied, mean=0.0, std=0.02)

    def param_count(self) -> int:
        """Parameters *owned* by this head (a tied decoder weight is not)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, h: Tensor) -> Tensor:
        """Project encoder states to vocabulary logits.

        Args:
            h: ``[B, L, d_model]``.

        Returns:
            ``[B, L, vocab]`` logits.
        """
        x = self.ln(F.gelu(self.dense(h)))
        return F.linear(x, self.decoder_weight.to(x.dtype), self.bias.to(x.dtype))
