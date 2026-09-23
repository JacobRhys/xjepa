"""Standalone throughput / efficiency benchmark for the x-JEPA training step.

What it measures
----------------
a. steps/sec and tokens/sec;
b. achieved model FLOP utilisation (MFU) against the card's dense bf16 peak;
c. **host-to-device transferred bytes per step** -- the number this repo exists to
   keep at zero (docs/CONTRACTS.md, "core efficiency invariant"). Measured from
   the CUDA profiler's memcpy events, and cross-checked with
   ``torch.cuda.memory_stats``;
d. the forward / backward / optimiser time split, from CUDA events;
e. the number of implicit device syncs, via
   ``torch.cuda.set_sync_debug_mode("warn")`` with the warnings captured.

Discipline
----------
* Warmup iterations are run and discarded; with ``--compile`` the compile step is
  guaranteed to fall inside warmup (the harness asserts a minimum warmup).
* Timing uses ``torch.cuda.Event`` pairs, with exactly one
  ``torch.cuda.synchronize()`` after the measured loop -- this file is the only
  place in the repo allowed to call it.
* Shapes are fixed for the whole run, matching the bucketing contract, so no
  recompilation happens inside the measured window.
* With no GPU present everything degrades to a clearly-labelled CPU-only run.

Run it::

    python -m xjepa.perf.bench --steps 50 --warmup 10 --batch-size 128 --seq-len 512
    python -m xjepa.perf.bench --model xjepa --compile --json out/bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BenchConfig",
    "BenchResult",
    "PEAK_BF16_TFLOPS",
    "SyntheticCorpus",
    "BenchModel",
    "peak_bf16_tflops",
    "transformer_flops_per_step",
    "run_benchmark",
    "main",
]


# --------------------------------------------------------------------------------------
# Hardware peaks (dense bf16, no sparsity, vendor spec sheets)
# --------------------------------------------------------------------------------------

#: Substring -> dense bf16 TFLOP/s. Matched case-insensitively against
#: ``torch.cuda.get_device_name()``. Values are *dense* (no 2:4 sparsity) and
#: for consumer Ada parts assume the fp32-accumulate rate, which is what a real
#: training step gets -- the doubled "with fp16 accumulate" marketing number is
#: not achievable here.
PEAK_BF16_TFLOPS: Dict[str, float] = {
    "4090": 165.2,
    "4080": 97.5,
    "3090": 71.0,
    "a5000": 111.0,
    "a6000": 155.0,
    "a100": 312.0,
    "h100": 989.0,
    "l4": 121.0,
    "l40": 181.0,
    "v100": 0.0,  # no bf16
    "t4": 0.0,  # no bf16
}


def peak_bf16_tflops(device_name: str) -> Optional[float]:
    """Look up a card's dense bf16 peak, or ``None`` if unknown/unsupported."""
    name = device_name.lower()
    for key, val in PEAK_BF16_TFLOPS.items():
        if key in name:
            return val if val > 0 else None
    return None


#: Reference parameter counts for the real model (measured from the faithful
#: ESM-2 t6 config: 6 layers, d_model 320, 20 heads, FFN 1280, RoPE -- so no
#: learned position table -- LayerNorm + GELU). "8M" is a round-up and using it
#: inflates reported MFU by ~8%.
#:
#: Which N to divide by is a *reporting decision*, not a detail: the JEPA
#: conditions carry ~0.69M extra predictor parameters, so an MFU table that mixes
#: them is not comparing like with like. Always state the N used.
REFERENCE_PARAMS: Dict[str, int] = {
    "encoder": 7_408_960,  # encoder alone -- the C1 baseline N for 6*N*D
    "c1_mlm_tied": 7_512_353,  # + tied MLM head
    "c2_c3_jepa": 8_099_968,  # + predictor (C2/C3/C4/C5c)
}


def transformer_flops_per_step(
    n_params_nonembedding: int,
    tokens_per_step: int,
    n_layers: int,
    d_model: int,
    seq_len: int,
) -> float:
    """FLOPs for one fwd+bwd step of a dense transformer.

    Uses the standard accounting (Kaplan et al. 2020 / Chowdhery et al. 2022):

        ``FLOPs/token = 6 * N_nonembed  +  12 * n_layers * d_model * seq_len``

    The second term is the attention score/context matmuls, which are *not*
    captured by the parameter count and which dominate more than people expect at
    this scale: with N = 7.41M and L = 512 they are **~21%** of total step FLOPs.
    Dropping them (the common "6ND is close enough" shortcut) understates the work
    done and therefore understates MFU by a fifth.

    Args:
        n_params_nonembedding: parameter count excluding token/position embeddings.
        tokens_per_step: batch_size * seq_len.
        n_layers: transformer depth.
        d_model: model width.
        seq_len: sequence length.

    Returns:
        FLOPs per optimiser step.
    """
    per_token = 6.0 * n_params_nonembedding + 12.0 * n_layers * d_model * seq_len
    return per_token * tokens_per_step


def detect_sdpa_backend(
    device: torch.device,
    batch: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    has_attn_mask: bool,
) -> str:
    """Which ``scaled_dot_product_attention`` backend these shapes will dispatch to.

    This matters for the MFU denominator's interpretation. We pass an arbitrary
    key-padding mask, and PyTorch's **flash** kernel accepts only ``is_causal``
    or no mask at all -- so the dispatch lands on **mem-efficient attention**
    (cutlassF), which is meaningfully slower than flash at L = 512. Assume the
    mem-efficient kernel when reasoning about the achievable fraction of peak,
    and treat this function's answer (plus the kernel names the profiler
    actually recorded) as the evidence rather than assuming either way.

    Args:
        device: target device.
        batch, n_heads, seq_len, head_dim: attention shapes.
        dtype: the dtype attention will run in (bf16 under autocast).
        has_attn_mask: whether a non-None ``attn_mask`` is passed.

    Returns:
        One of ``"flash"``, ``"mem_efficient"``, ``"math"``, ``"cpu"`` or
        ``"unknown: <reason>"``.
    """
    if device.type != "cuda":
        return "cpu (no CUDA SDPA dispatch)"
    try:
        shape = (batch, n_heads, seq_len, head_dim)
        q = torch.empty(shape, device=device, dtype=dtype)
        mask = (
            torch.zeros(batch, 1, 1, seq_len, device=device, dtype=dtype)
            if has_attn_mask
            else None
        )
        params_cls = getattr(torch.backends.cuda, "SDPAParams", None)
        if params_cls is None:
            return "unknown: torch.backends.cuda.SDPAParams unavailable"
        try:
            params = params_cls(q, q, q, mask, 0.0, False, False)
        except TypeError:
            params = params_cls(q, q, q, mask, 0.0, False)
        if torch.backends.cuda.can_use_flash_attention(params, False):
            return "flash"
        if torch.backends.cuda.can_use_efficient_attention(params, False):
            return "mem_efficient"
        return "math"
    except Exception as exc:  # noqa: BLE001
        return f"unknown: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------
# Config / result
# --------------------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """Benchmark parameters. Defaults mirror RESEARCH_PLAN.md 3.1/3.3."""

    steps: int = 50
    warmup: int = 10
    batch_size: int = 128
    seq_len: int = 512
    n_layers: int = 6
    d_model: int = 320
    n_heads: int = 20
    d_ff: int = 1280
    vocab: int = 33
    target_dim: int = 128
    corpus_seqs: int = 2048
    device: Optional[str] = None
    compile: bool = False
    model: str = "synthetic"  # "synthetic" | "xjepa"
    seed: int = 0
    measure_transfers: bool = True
    transfer_steps: int = 5
    autocast: bool = True
    #: Whether a key-padding mask is passed to attention. ``True`` mirrors the
    #: ``hybrid`` bucket policy (some padding, so a mask is required, so SDPA
    #: can only reach cutlassF). ``False`` mirrors ``crop``, which leaves zero
    #: padding and passes no mask at all -- the only route to the flash backend,
    #: since PyTorch's flash path accepts only ``is_causal`` or no mask.
    attn_mask: bool = True
    #: Which parameter count feeds ``6*N*D``. ``"measured"`` uses the built
    #: model's non-embedding count; any key of :data:`REFERENCE_PARAMS`
    #: ("encoder", "c1_mlm_tied", "c2_c3_jepa") pins it to the real model's
    #: count so numbers stay comparable when the benchmark drives a stand-in.
    params_basis: str = "measured"

    def resolved_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len


@dataclass
class BenchResult:
    """Everything the harness measured, plus enough provenance to reproduce it."""

    config: Dict[str, Any]
    device_name: str
    torch_version: str
    cuda_available: bool
    n_params: int
    n_params_nonembedding: int

    steps_per_sec: float
    tokens_per_sec: float
    step_ms_mean: float
    step_ms_p50: float
    step_ms_p90: float

    fwd_ms: float
    bwd_ms: float
    opt_ms: float
    other_ms: float

    flops_per_step: float
    achieved_tflops: float
    flops_params_used: int = 0
    params_basis: str = "measured"
    sdpa_backend: str = "unknown"
    sdpa_kernels_observed: List[str] = field(default_factory=list)
    peak_bf16_tflops: Optional[float] = None
    mfu: Optional[float] = None

    h2d_bytes_per_step: Optional[float] = None
    d2h_bytes_per_step: Optional[float] = None
    h2d_copies_per_step: Optional[float] = None
    transfer_measurement: str = "not attempted"

    implicit_syncs: Optional[int] = None
    sync_messages: List[str] = field(default_factory=list)
    sync_measurement: str = "not attempted"

    peak_memory_gb: Optional[float] = None
    corpus_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render as a reviewer-readable markdown block."""
        mfu = f"{self.mfu * 100:.1f}%" if self.mfu is not None else "n/a"
        peak = f"{self.peak_bf16_tflops:.0f}" if self.peak_bf16_tflops else "unknown"
        h2d = (
            f"{self.h2d_bytes_per_step:,.0f} B"
            if self.h2d_bytes_per_step is not None
            else "not measured"
        )
        d2h = (
            f"{self.d2h_bytes_per_step:,.0f} B"
            if self.d2h_bytes_per_step is not None
            else "not measured"
        )
        syncs = str(self.implicit_syncs) if self.implicit_syncs is not None else "not measured"
        total = max(self.step_ms_mean, 1e-9)
        lines = [
            "# x-JEPA training-step benchmark",
            "",
            f"- device: **{self.device_name}** (torch {self.torch_version}, "
            f"cuda available: {self.cuda_available})",
            f"- model: {self.n_params:,} params "
            f"({self.n_params_nonembedding:,} non-embedding)",
            f"- shape: batch {self.config['batch_size']} x seq {self.config['seq_len']} "
            f"= {self.config['batch_size'] * self.config['seq_len']:,} tokens/step",
            f"- resident synthetic corpus: {self.corpus_bytes / 1e9:.2f} GB",
            "",
            "## Throughput",
            "",
            "| metric | value |",
            "|---|---|",
            f"| steps/sec | {self.steps_per_sec:.2f} |",
            f"| tokens/sec | {self.tokens_per_sec:,.0f} |",
            f"| step time mean / p50 / p90 (ms) | {self.step_ms_mean:.2f} / "
            f"{self.step_ms_p50:.2f} / {self.step_ms_p90:.2f} |",
            f"| achieved TFLOP/s | {self.achieved_tflops:.2f} |",
            f"| card dense bf16 peak (TFLOP/s) | {peak} |",
            f"| **MFU** | **{mfu}** |",
            f"| N used in 6*N*D | {self.flops_params_used:,} ({self.params_basis}) |",
            f"| SDPA backend (predicted) | {self.sdpa_backend} |",
            f"| SDPA kernels observed | "
            f"{', '.join(self.sdpa_kernels_observed) if self.sdpa_kernels_observed else 'n/a'} |",
            "",
            "## Step time split",
            "",
            "| phase | ms | share |",
            "|---|---|---|",
            f"| forward | {self.fwd_ms:.2f} | {100 * self.fwd_ms / total:.1f}% |",
            f"| backward | {self.bwd_ms:.2f} | {100 * self.bwd_ms / total:.1f}% |",
            f"| optimiser | {self.opt_ms:.2f} | {100 * self.opt_ms / total:.1f}% |",
            f"| other (batching, launch gaps) | {self.other_ms:.2f} | "
            f"{100 * self.other_ms / total:.1f}% |",
            "",
            "## Data movement and stalls",
            "",
            "| metric | value |",
            "|---|---|",
            f"| host->device bytes / step | {h2d} |",
            f"| device->host bytes / step | {d2h} |",
            f"| H2D copies / step | "
            f"{self.h2d_copies_per_step if self.h2d_copies_per_step is not None else 'n/a'} |",
            f"| transfer measurement | {self.transfer_measurement} |",
            f"| implicit device syncs (measured window) | {syncs} |",
            f"| sync measurement | {self.sync_measurement} |",
            f"| peak allocated memory | "
            f"{f'{self.peak_memory_gb:.2f} GB' if self.peak_memory_gb is not None else 'n/a'} |",
        ]
        if self.sync_messages:
            lines += ["", "### Sync warnings (deduplicated)", ""]
            lines += [f"- `{m}`" for m in self.sync_messages[:10]]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Resident corpus + model
# --------------------------------------------------------------------------------------


class SyntheticCorpus:
    """A whole 'corpus' resident on the device, mimicking ``data/store.GpuCorpus``.

    Exists so the benchmark can prove the zero-transfer property without
    depending on the real data module: tokens and targets are allocated once on
    device and every batch is produced by *index arithmetic on device*. The only
    host->device traffic possible per step is the step counter, and we do not even
    do that -- the index tensor is generated by a device-side RNG.

    Args:
        n_seqs: number of sequences held resident.
        seq_len: fixed sequence length (bucketed shapes in the real pipeline).
        vocab: token vocabulary size.
        target_dim: PCA-reduced structure target width.
        device: device to allocate on.
        seed: RNG seed for the contents.
    """

    def __init__(
        self,
        n_seqs: int,
        seq_len: int,
        vocab: int,
        target_dim: int,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        gen = torch.Generator(device=device.type if device.type != "mps" else "cpu")
        gen.manual_seed(seed)
        self.device = device
        self.seq_len = seq_len
        self.tokens = torch.randint(
            0, vocab, (n_seqs, seq_len), device=device, dtype=torch.uint8, generator=gen
        )
        self.targets = torch.randn(
            (n_seqs, seq_len, target_dim), device=device, dtype=torch.float32, generator=gen
        ).to(torch.float16)
        self.n_seqs = n_seqs

    @property
    def nbytes(self) -> int:
        return self.tokens.numel() * self.tokens.element_size() + (
            self.targets.numel() * self.targets.element_size()
        )

    def batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw a fixed-shape batch. All index math happens on device.

        Returns:
            ``(tokens int64 [B, L], targets fp16 [B, L, T], pad_mask bool [B, L],
            mask_sel bool [B, L])``.
        """
        idx = torch.randint(0, self.n_seqs, (batch_size,), device=self.device)
        tokens = self.tokens.index_select(0, idx).long()
        targets = self.targets.index_select(0, idx)
        pad_mask = torch.ones(batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        mask_sel = torch.rand(batch_size, self.seq_len, device=self.device) < 0.15
        return tokens, targets, pad_mask, mask_sel


class _Block(nn.Module):
    """Pre-LN transformer block; attention strictly via SDPA (contract rule 5)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, l, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(b, l, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(o.transpose(1, 2).reshape(b, l, d))
        return x + self.ff(self.ln2(x))


class BenchModel(nn.Module):
    """Encoder + linear structure-prediction head, sized to the ESM-2 t6 8M config.

    This is a stand-in for ``xjepa.model.encoder.Encoder`` + ``heads.Predictor``
    with identical shapes and the same attention path, so that FLOP accounting
    and the measured step time transfer to the real model. The benchmark can also
    drive the real modules (``--model xjepa``) once they land.
    """

    def __init__(self, cfg: BenchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.blocks = nn.ModuleList(
            [_Block(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.target_dim)

    def n_params_nonembedding(self) -> int:
        emb = self.embed.weight.numel() + self.pos.weight.numel()
        return sum(p.numel() for p in self.parameters()) - emb

    def forward(
        self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor], mask_sel: torch.Tensor
    ) -> torch.Tensor:
        b, l = tokens.shape
        x = self.embed(tokens)
        x = torch.where(mask_sel.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = x + self.pos.weight[:l].unsqueeze(0)
        # Key-padding mask, broadcast rather than materialised, and built without
        # ever asking the host whether padding exists (`pad_mask.all()` would be a
        # device sync -- exactly what this harness is here to catch).
        # `None` means the batch has zero padding (the `crop` policy); passing an
        # all-true mask instead would silently forfeit the flash backend.
        attn_mask = None if pad_mask is None else pad_mask[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.head(self.ln_f(x))


def _build_model(cfg: BenchConfig, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Build the benchmark model, optionally the real one, with clear fallback."""
    notes: List[str] = []
    if cfg.model == "xjepa":
        try:  # pragma: no cover - depends on a module owned by another agent
            from xjepa.model.encoder import Encoder, EncoderConfig  # type: ignore

            enc_cfg = EncoderConfig(
                n_layers=cfg.n_layers,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                vocab=cfg.vocab,
                max_len=cfg.seq_len,
            )
            encoder = Encoder(enc_cfg)

            class _Wrapped(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.encoder = encoder
                    self.head = nn.Linear(cfg.d_model, cfg.target_dim)

                def n_params_nonembedding(self) -> int:
                    total = sum(p.numel() for p in self.parameters())
                    emb = sum(
                        m.weight.numel()
                        for m in self.modules()
                        if isinstance(m, nn.Embedding)
                    )
                    return total - emb

                def forward(
                    self,
                    tokens: torch.Tensor,
                    pad_mask: Optional[torch.Tensor],
                    mask_sel: torch.Tensor,
                ) -> torch.Tensor:
                    return self.head(self.encoder(tokens, pad_mask))

            notes.append("Driving the real xjepa.model.encoder.Encoder.")
            return _Wrapped().to(device), notes
        except Exception as exc:  # noqa: BLE001 - we want any import/API failure here
            notes.append(
                f"--model xjepa requested but unusable ({type(exc).__name__}: {exc}); "
                "fell back to the shape-identical synthetic model."
            )
    return BenchModel(cfg).to(device), notes


# --------------------------------------------------------------------------------------
# Instrumentation helpers
# --------------------------------------------------------------------------------------


class _PhaseClock:
    """Per-phase timing that uses CUDA events on GPU and perf_counter on CPU."""

    def __init__(self, device: torch.device, n_steps: int) -> None:
        self.cuda = device.type == "cuda"
        self.n_steps = n_steps
        self._marks: List[List[Any]] = []
        self._cur: List[Any] = []

    def mark(self) -> None:
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._cur.append(ev)
        else:
            self._cur.append(time.perf_counter())

    def end_step(self) -> None:
        self._marks.append(self._cur)
        self._cur = []

    def elapsed_ms(self) -> List[List[float]]:
        """Return per-step ``[t_fwd, t_bwd, t_opt, t_total]`` in milliseconds."""
        if self.cuda:
            torch.cuda.synchronize()  # the single allowed sync, after the loop
        out: List[List[float]] = []
        for marks in self._marks:
            if len(marks) != 4:
                continue
            if self.cuda:
                spans = [marks[i].elapsed_time(marks[i + 1]) for i in range(3)]
                total = marks[0].elapsed_time(marks[3])
            else:
                spans = [(marks[i + 1] - marks[i]) * 1e3 for i in range(3)]
                total = (marks[3] - marks[0]) * 1e3
            out.append([spans[0], spans[1], spans[2], total])
        return out


@contextmanager
def sync_debug_capture(enabled: bool) -> Iterator[List[str]]:
    """Capture implicit-sync warnings raised by ``set_sync_debug_mode('warn')``.

    PyTorch emits a Python ``UserWarning`` from the dispatcher every time an
    operation forces a device synchronisation (``.item()``, ``.cpu()``,
    ``nonzero()``, a host-visible ``bool()`` on a GPU tensor ...). That is exactly
    the class of mistake hard rule 1 in docs/CONTRACTS.md forbids, so we turn the
    warning on around the measured loop and collect what comes out.

    Args:
        enabled: pass False (CPU, or unsupported build) to no-op.

    Yields:
        The list that will hold the captured warning messages.
    """
    messages: List[str] = []
    if not enabled:
        yield messages
        return
    prev = None
    try:
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("warn")
    except Exception:  # noqa: BLE001 - old/CPU builds
        yield messages
        return
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield messages
        for w in caught:
            text = str(w.message)
            if "sync" in text.lower() or "synchroniz" in text.lower():
                messages.append(text)
    finally:
        try:
            torch.cuda.set_sync_debug_mode(prev if prev is not None else 0)
        except Exception:  # noqa: BLE001
            pass


def _measure_transfers(
    step_fn: Callable[[], None], n_steps: int, device: torch.device, trace_dir: Path
) -> Dict[str, Any]:
    """Profile ``n_steps`` steps and total the memcpy bytes crossing the PCIe bus.

    Bytes are read from the kineto chrome trace, where every ``Memcpy HtoD`` /
    ``Memcpy DtoH`` event carries ``args["bytes"]``. The profiler is the only
    source that reports this; ``torch.cuda.memory_stats`` counts allocations, not
    transfers, so we report it alongside as corroboration, not as the number.

    Args:
        step_fn: closure running exactly one training step.
        n_steps: how many steps to profile.
        device: the benchmark device.
        trace_dir: where the chrome trace is written.

    Returns:
        Dict with ``h2d_bytes``, ``d2h_bytes``, ``h2d_count``, ``d2h_count``,
        ``method`` and ``ok``.
    """
    out: Dict[str, Any] = {
        "h2d_bytes": None,
        "d2h_bytes": None,
        "h2d_count": None,
        "d2h_count": None,
        "method": "unavailable",
        "ok": False,
    }
    if device.type != "cuda":
        out["method"] = "skipped: CPU-only run has no host-device boundary"
        return out
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        out["method"] = f"torch.profiler unavailable ({exc})"
        return out

    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "xjepa_bench_trace.json"
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
        ) as prof:
            for _ in range(n_steps):
                # An nvtx range so the same window is identifiable under nsys/ncu.
                torch.cuda.nvtx.range_push("xjepa_bench_step")
                step_fn()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text()).get("traceEvents", [])
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        out["method"] = f"profiling failed ({type(exc).__name__}: {exc})"
        return out

    h2d_bytes = d2h_bytes = 0
    h2d_count = d2h_count = 0
    attn_kernels: List[str] = []
    for ev in events:
        name = str(ev.get("name", ""))
        low_all = name.lower()
        if any(tag in low_all for tag in ("fmha", "flash", "efficient_attention", "attention_kernel")):
            if name not in attn_kernels and len(attn_kernels) < 5:
                attn_kernels.append(name)
        if "memcpy" not in low_all:
            continue
        nbytes = int(ev.get("args", {}).get("bytes", 0) or 0)
        low = name.lower()
        if "htod" in low:
            h2d_bytes += nbytes
            h2d_count += 1
        elif "dtoh" in low:
            d2h_bytes += nbytes
            d2h_count += 1
    out.update(
        h2d_bytes=h2d_bytes / n_steps,
        d2h_bytes=d2h_bytes / n_steps,
        h2d_count=h2d_count / n_steps,
        d2h_count=d2h_count / n_steps,
        method=f"kineto memcpy events over {n_steps} steps (trace: {trace_path})",
        attn_kernels=attn_kernels,
        ok=True,
    )
    return out


# --------------------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------------------


def run_benchmark(cfg: BenchConfig, verbose: bool = True) -> BenchResult:
    """Run the full benchmark and return the result.

    Args:
        cfg: benchmark configuration.
        verbose: print progress/degradation messages to stderr.

    Returns:
        A populated :class:`BenchResult`. On a machine without CUDA every
        GPU-only field is ``None`` and the reason is recorded in ``notes``.
    """
    device = cfg.resolved_device()
    is_cuda = device.type == "cuda"
    notes: List[str] = []
    torch.manual_seed(cfg.seed)

    def log(msg: str) -> None:
        if verbose:
            print(msg, file=sys.stderr)

    if not is_cuda:
        log(
            "[bench] No CUDA device available -- running CPU-only.\n"
            "        steps/sec, tokens/sec and the fwd/bwd/opt split are still "
            "measured (wall clock).\n"
            "        MFU, host->device transfer bytes and implicit-sync counts "
            "are GPU-only and will be reported as unavailable."
        )
        notes.append(
            "CPU-only run: MFU, transfer bytes and sync counts are unavailable by "
            "construction (no host-device boundary, no vendor bf16 peak)."
        )

    corpus = SyntheticCorpus(
        cfg.corpus_seqs, cfg.seq_len, cfg.vocab, cfg.target_dim, device, seed=cfg.seed
    )
    model, build_notes = _build_model(cfg, device)
    notes.extend(build_notes)

    # NOTE (contract problem): docs/CONTRACTS.md hard rule 4 asks for
    # "foreach=True, fused=True" on AdamW. PyTorch rejects that combination --
    # `RuntimeError: \`fused\` and \`foreach\` cannot be \`True\` together.` The
    # intended optimiser is fused on CUDA (it is the faster of the two and
    # subsumes the foreach path) and foreach on CPU, which is what we use here.
    fused_ok = is_cuda
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        foreach=not fused_ok,
        fused=fused_ok,
    )
    notes.append(
        "AdamW: fused=True on CUDA / foreach=True on CPU. Contract rule 4 asks "
        "for both flags True, which PyTorch refuses -- they are mutually "
        "exclusive. Rule 4 needs amending."
    )

    step_model: nn.Module = model
    if cfg.compile:
        if cfg.warmup < 5:
            raise ValueError("--compile requires warmup >= 5 so the compile step is discarded")
        step_model = torch.compile(model, dynamic=False)  # fixed shapes, contract rule 3
        notes.append("torch.compile enabled (dynamic=False); compile cost absorbed by warmup.")

    amp_dtype = torch.bfloat16
    use_amp = cfg.autocast and is_cuda and torch.cuda.is_bf16_supported()
    if cfg.autocast and not use_amp:
        notes.append("bf16 autocast disabled (not supported on this device); running fp32.")

    clock = _PhaseClock(device, cfg.steps)

    def one_step(timed: bool) -> None:
        tokens, targets, pad_mask, mask_sel = corpus.batch(cfg.batch_size)
        if not cfg.attn_mask:
            # `crop` policy: zero padding, so no mask is built at all.
            pad_mask = None
        if timed:
            clock.mark()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = step_model(tokens, pad_mask, mask_sel)
            loss = F.smooth_l1_loss(
                pred.float()[mask_sel], targets.float()[mask_sel], beta=1.0
            )
        if timed:
            clock.mark()
        loss.backward()
        if timed:
            clock.mark()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if timed:
            clock.mark()
            clock.end_step()

    # ---- warmup (compile + autotune + allocator warmup), discarded -------------------
    log(f"[bench] warmup: {cfg.warmup} steps on {device}")
    for _ in range(cfg.warmup):
        one_step(timed=False)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ---- measured loop ---------------------------------------------------------------
    log(f"[bench] measuring: {cfg.steps} steps")
    t0 = time.perf_counter()
    with sync_debug_capture(is_cuda) as sync_msgs:
        for _ in range(cfg.steps):
            one_step(timed=True)
        if is_cuda:
            torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    spans = clock.elapsed_ms()
    if not spans:
        raise RuntimeError("no steps were timed")
    fwd = statistics.mean(s[0] for s in spans)
    bwd = statistics.mean(s[1] for s in spans)
    optm = statistics.mean(s[2] for s in spans)
    totals = sorted(s[3] for s in spans)
    step_mean = statistics.mean(totals)
    p50 = totals[len(totals) // 2]
    p90 = totals[min(len(totals) - 1, int(0.9 * len(totals)))]

    steps_per_sec = cfg.steps / wall
    tokens_per_sec = steps_per_sec * cfg.tokens_per_step

    n_params = sum(p.numel() for p in model.parameters())
    n_nonemb = (
        model.n_params_nonembedding()
        if hasattr(model, "n_params_nonembedding")
        else n_params
    )
    if cfg.params_basis == "measured":
        flops_params = n_nonemb
    elif cfg.params_basis in REFERENCE_PARAMS:
        flops_params = REFERENCE_PARAMS[cfg.params_basis]
        notes.append(
            f"MFU uses the reference N for '{cfg.params_basis}' "
            f"({flops_params:,}) rather than the built model's non-embedding "
            f"count ({n_nonemb:,})."
        )
    else:
        raise ValueError(
            f"params_basis must be 'measured' or one of {sorted(REFERENCE_PARAMS)}"
        )
    notes.append(
        "MFU denominator uses N = "
        f"{flops_params:,}. The real encoder is 7,408,960 params; with the tied "
        "MLM head C1 is 7,512,353 and with the predictor C2/C3 are 8,099,968 -- "
        "state which N a cross-condition MFU table used, the spread is ~9%."
    )
    flops_step = transformer_flops_per_step(
        flops_params, cfg.tokens_per_step, cfg.n_layers, cfg.d_model, cfg.seq_len
    )
    achieved_tflops = flops_step * steps_per_sec / 1e12

    device_name = torch.cuda.get_device_name(device) if is_cuda else platform.processor() or "CPU"
    peak = peak_bf16_tflops(device_name) if is_cuda else None
    mfu = achieved_tflops / peak if peak else None
    if is_cuda and peak is None:
        notes.append(
            f"No bf16 peak on file for '{device_name}' -- add it to PEAK_BF16_TFLOPS "
            "to get an MFU number."
        )

    # ---- attention backend -----------------------------------------------------------
    sdpa_backend = detect_sdpa_backend(
        device,
        batch=cfg.batch_size,
        n_heads=cfg.n_heads,
        seq_len=cfg.seq_len,
        head_dim=cfg.d_model // cfg.n_heads,
        dtype=amp_dtype if use_amp else torch.float32,
        has_attn_mask=cfg.attn_mask,
    )
    if sdpa_backend.startswith("mem_efficient"):
        notes.append(
            "Attention dispatches to the mem-efficient (cutlassF) kernel, not "
            "flash: we pass an arbitrary additive key-padding mask and the flash "
            "path accepts only is_causal or no mask. Budget peak accordingly -- "
            "mem-efficient is materially slower than flash at L=512."
        )

    # ---- transfers -------------------------------------------------------------------
    transfer: Dict[str, Any] = {"method": "not attempted", "ok": False}
    if cfg.measure_transfers:
        trace_dir = Path(".xjepa_bench")
        transfer = _measure_transfers(
            lambda: one_step(timed=False), cfg.transfer_steps, device, trace_dir
        )

    # ---- syncs -----------------------------------------------------------------------
    seen: List[str] = []
    for m in sync_msgs:
        if m not in seen:
            seen.append(m)
    sync_count: Optional[int] = len(sync_msgs) if is_cuda else None
    sync_method = (
        "torch.cuda.set_sync_debug_mode('warn') with warnings captured"
        if is_cuda
        else "skipped: CPU-only run cannot have implicit device syncs"
    )

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else None

    return BenchResult(
        config=asdict(cfg),
        device_name=device_name,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        n_params=n_params,
        n_params_nonembedding=n_nonemb,
        steps_per_sec=steps_per_sec,
        tokens_per_sec=tokens_per_sec,
        step_ms_mean=step_mean,
        step_ms_p50=p50,
        step_ms_p90=p90,
        fwd_ms=fwd,
        bwd_ms=bwd,
        opt_ms=optm,
        other_ms=max(0.0, step_mean - (fwd + bwd + optm)),
        flops_per_step=flops_step,
        achieved_tflops=achieved_tflops,
        flops_params_used=int(flops_params),
        params_basis=cfg.params_basis,
        sdpa_backend=sdpa_backend,
        sdpa_kernels_observed=list(transfer.get("attn_kernels") or []),
        peak_bf16_tflops=peak,
        mfu=mfu,
        h2d_bytes_per_step=transfer.get("h2d_bytes"),
        d2h_bytes_per_step=transfer.get("d2h_bytes"),
        h2d_copies_per_step=transfer.get("h2d_count"),
        transfer_measurement=str(transfer.get("method")),
        implicit_syncs=sync_count,
        sync_messages=seen,
        sync_measurement=sync_method,
        peak_memory_gb=peak_mem,
        corpus_bytes=corpus.nbytes,
        notes=notes,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for ``python -m xjepa.perf.bench``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = BenchConfig()
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--warmup", type=int, default=d.warmup)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--seq-len", type=int, default=d.seq_len)
    p.add_argument("--n-layers", type=int, default=d.n_layers)
    p.add_argument("--d-model", type=int, default=d.d_model)
    p.add_argument("--n-heads", type=int, default=d.n_heads)
    p.add_argument("--d-ff", type=int, default=d.d_ff)
    p.add_argument("--corpus-seqs", type=int, default=d.corpus_seqs)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--model", choices=("synthetic", "xjepa"), default=d.model)
    p.add_argument("--compile", action="store_true", help="wrap the step in torch.compile")
    p.add_argument("--no-transfers", action="store_true", help="skip the profiler transfer pass")
    p.add_argument(
        "--params-basis",
        choices=("measured",) + tuple(sorted(REFERENCE_PARAMS)),
        default=d.params_basis,
        help="which N feeds 6*N*D in the MFU calculation",
    )
    p.add_argument("--json", type=str, default=None, help="write the raw result as JSON here")
    p.add_argument("--markdown", type=str, default=None, help="write the markdown report here")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    cfg = BenchConfig(
        steps=args.steps,
        warmup=args.warmup,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        corpus_seqs=args.corpus_seqs,
        device=args.device,
        compile=args.compile,
        model=args.model,
        measure_transfers=not args.no_transfers,
        params_basis=args.params_basis,
    )
    result = run_benchmark(cfg, verbose=not args.quiet)
    report = result.to_markdown()
    print(report)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2, default=str))
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
