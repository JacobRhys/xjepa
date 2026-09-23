"""Short profiling run -> readable markdown report.

Answers the three questions a reviewer asks about a training loop:

1. **Where does the time actually go?** Top kernels/ops by self time.
2. **Is anything copying across PCIe in the hot loop?** Every ``Memcpy HtoD`` /
   ``DtoH`` event in the profiled window, with its byte count and the stack-free
   name so it is obvious which one to kill.
3. **Is torch.compile re-tracing?** Dynamo frame/recompile counters plus any
   "recompiling" log lines captured from the ``torch._dynamo`` logger.

Usage::

    python -m xjepa.perf.profile_report --steps 8 --out docs/profile_latest.md
    python -m xjepa.perf.profile_report --compile --batch-size 128 --seq-len 512
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .bench import BenchConfig, SyntheticCorpus, _build_model

__all__ = ["OpStat", "CopyStat", "ProfileReport", "run_profile", "main"]


@dataclass
class OpStat:
    """One row of the "top stalls" table."""

    name: str
    self_time_us: float
    total_time_us: float
    count: int
    device: str

    @property
    def self_time_ms(self) -> float:
        return self.self_time_us / 1e3


@dataclass
class CopyStat:
    """One host<->device copy observed inside the profiled window."""

    name: str
    count: int
    total_bytes: int
    direction: str


@dataclass
class ProfileReport:
    """Everything the profiling run found."""

    device: str
    steps: int
    compiled: bool
    config: Dict[str, Any]
    top_ops: List[OpStat] = field(default_factory=list)
    copies: List[CopyStat] = field(default_factory=list)
    recompile_messages: List[str] = field(default_factory=list)
    dynamo_counters: Dict[str, Any] = field(default_factory=dict)
    total_self_time_us: float = 0.0
    notes: List[str] = field(default_factory=list)
    trace_path: Optional[str] = None

    def to_markdown(self) -> str:
        """Render the report."""
        h2d = sum(c.total_bytes for c in self.copies if c.direction == "HtoD")
        d2h = sum(c.total_bytes for c in self.copies if c.direction == "DtoH")
        lines = [
            "# x-JEPA profile report",
            "",
            f"- device: **{self.device}**, {self.steps} profiled steps, "
            f"torch.compile: {'on' if self.compiled else 'off'}",
            f"- shape: batch {self.config.get('batch_size')} x seq "
            f"{self.config.get('seq_len')}",
            f"- total profiled self time: {self.total_self_time_us / 1e3:.1f} ms",
        ]
        if self.trace_path:
            lines.append(f"- chrome trace: `{self.trace_path}`")
        lines += ["", "## Top stalls (by self time)", ""]
        if self.top_ops:
            lines += [
                "| # | op | device | self ms | share | calls |",
                "|---|---|---|---|---|---|",
            ]
            denom = max(self.total_self_time_us, 1e-9)
            for i, op in enumerate(self.top_ops, 1):
                lines.append(
                    f"| {i} | `{op.name}` | {op.device} | {op.self_time_ms:.3f} | "
                    f"{100 * op.self_time_us / denom:.1f}% | {op.count} |"
                )
        else:
            lines.append("_No operator events were recorded._")

        lines += ["", "## Host <-> device copies in the hot loop", ""]
        if self.copies:
            lines += ["| name | direction | count | bytes | bytes/step |", "|---|---|---|---|---|"]
            for c in self.copies:
                lines.append(
                    f"| `{c.name}` | {c.direction} | {c.count} | {c.total_bytes:,} | "
                    f"{c.total_bytes / max(self.steps, 1):,.0f} |"
                )
            lines += [
                "",
                f"**Totals:** {h2d:,} B host->device, {d2h:,} B device->host over "
                f"{self.steps} steps "
                f"({h2d / max(self.steps, 1):,.0f} B/step H2D).",
            ]
            if h2d > 0:
                lines.append(
                    "> Non-zero H2D traffic in the training loop violates the core "
                    "efficiency invariant in docs/CONTRACTS.md. Find the tensor that "
                    "is being created on the host (a Python scalar turned into a "
                    "tensor, a mask built with `torch.tensor(...)`, an index list) "
                    "and allocate it on device once, outside the loop."
                )
        elif self.device == "cpu":
            lines.append(
                "_Not applicable: a CPU-only run has no host-device boundary. "
                "Re-run this report on the target GPU to audit the hot loop._"
            )
        else:
            lines.append(
                "**None observed.** The corpus is resident in VRAM and batching is "
                "index arithmetic on device, so the steady-state loop crosses the "
                "PCIe bus zero times per step."
            )

        lines += ["", "## torch.compile / graph recompiles", ""]
        if not self.compiled:
            lines.append("_Not compiled for this run (`--compile` to check recompiles)._")
        elif self.recompile_messages:
            lines.append("Recompiles detected -- every one of these is a multi-second stall:")
            lines += [f"- `{m}`" for m in self.recompile_messages[:20]]
            lines.append(
                "> The usual cause here is a shape that is not one of the four "
                "buckets (128/256/384/512). Fixed shapes per bucket, "
                "`dynamic=False`, and nothing shape-dependent in Python."
            )
        else:
            lines.append(
                "No recompiles observed in the profiled window (shapes were fixed "
                "and warmup absorbed the initial compile)."
            )
        if self.dynamo_counters:
            lines += ["", "```", str(self.dynamo_counters), "```"]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


class _RecompileLogCapture(logging.Handler):
    """Collect ``torch._dynamo`` log records that mention recompilation."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001  # pragma: no cover
            return
        low = msg.lower()
        if "recompil" in low or "cache_size_limit" in low or "graph break" in low:
            self.messages.append(msg.replace("\n", " ")[:300])


def _classify_copy(name: str) -> Optional[str]:
    low = name.lower()
    if "memcpy" not in low:
        return None
    if "htod" in low:
        return "HtoD"
    if "dtoh" in low:
        return "DtoH"
    if "dtod" in low:
        return "DtoD"
    return "other"


def run_profile(
    cfg: BenchConfig,
    steps: int = 8,
    warmup: int = 5,
    top_k: int = 15,
    trace_path: Optional[str] = None,
) -> ProfileReport:
    """Profile a handful of training steps and summarise them.

    Args:
        cfg: the benchmark configuration (model shape, device, compile flag).
        steps: profiled steps (keep small -- the profiler has real overhead).
        warmup: unprofiled steps first, so compile and allocator warmup are
            excluded from the report.
        top_k: rows in the "top stalls" table.
        trace_path: optional path for a chrome trace export.

    Returns:
        A :class:`ProfileReport`.
    """
    from torch.profiler import ProfilerActivity, profile

    device = cfg.resolved_device()
    is_cuda = device.type == "cuda"
    notes: List[str] = []
    torch.manual_seed(cfg.seed)

    corpus = SyntheticCorpus(
        cfg.corpus_seqs, cfg.seq_len, cfg.vocab, cfg.target_dim, device, seed=cfg.seed
    )
    model, build_notes = _build_model(cfg, device)
    notes.extend(build_notes)
    # fused and foreach are mutually exclusive in PyTorch; see the note in
    # bench.run_benchmark about docs/CONTRACTS.md hard rule 4.
    opt = torch.optim.AdamW(
        model.parameters(), lr=4e-4, betas=(0.9, 0.98), weight_decay=0.01,
        foreach=not is_cuda, fused=is_cuda,
    )
    step_model = torch.compile(model, dynamic=False) if cfg.compile else model
    use_amp = cfg.autocast and is_cuda and torch.cuda.is_bf16_supported()
    if not is_cuda:
        notes.append(
            "CPU-only profile: rows below are CPU operators. Host<->device copy "
            "and MFU analysis require a GPU."
        )

    def one_step() -> None:
        tokens, targets, pad_mask, mask_sel = corpus.batch(cfg.batch_size)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = step_model(tokens, pad_mask, mask_sel)
            loss = F.smooth_l1_loss(pred.float()[mask_sel], targets.float()[mask_sel], beta=1.0)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Recompile capture has to be installed before the profiled window, and dynamo
    # counters reset, so that warmup's legitimate first compile is not reported.
    handler = _RecompileLogCapture()
    dynamo_logger = logging.getLogger("torch._dynamo")
    prev_level = dynamo_logger.level
    dynamo_logger.addHandler(handler)
    dynamo_logger.setLevel(logging.DEBUG)

    for _ in range(warmup):
        one_step()
    if is_cuda:
        torch.cuda.synchronize()
    handler.messages.clear()

    counters_before: Dict[str, Any] = {}
    try:
        from torch._dynamo.utils import counters as dynamo_counters

        counters_before = {k: dict(v) for k, v in dynamo_counters.items()}
    except Exception:  # noqa: BLE001  # pragma: no cover
        dynamo_counters = None  # type: ignore[assignment]

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if is_cuda else [])
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(steps):
            one_step()
        if is_cuda:
            torch.cuda.synchronize()

    dynamo_logger.removeHandler(handler)
    dynamo_logger.setLevel(prev_level)

    # --- top ops -----------------------------------------------------------------
    ops: List[OpStat] = []
    for ev in prof.key_averages():
        cuda_self = float(
            getattr(ev, "self_device_time_total", 0.0) or getattr(ev, "self_cuda_time_total", 0.0) or 0.0
        )
        cpu_self = float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0)
        if is_cuda and cuda_self > 0:
            self_us, dev = cuda_self, "cuda"
            total_us = float(
                getattr(ev, "device_time_total", 0.0) or getattr(ev, "cuda_time_total", 0.0) or 0.0
            )
        else:
            self_us, dev = cpu_self, "cpu"
            total_us = float(getattr(ev, "cpu_time_total", 0.0) or 0.0)
        if self_us <= 0:
            continue
        ops.append(
            OpStat(
                name=str(ev.key),
                self_time_us=self_us,
                total_time_us=total_us,
                count=int(getattr(ev, "count", 0) or 0),
                device=dev,
            )
        )
    ops.sort(key=lambda o: o.self_time_us, reverse=True)
    total_self = sum(o.self_time_us for o in ops)

    # --- copies ------------------------------------------------------------------
    copies: Dict[Tuple[str, str], CopyStat] = {}
    written_trace = None
    if is_cuda:
        path = Path(trace_path or ".xjepa_bench/profile_trace.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            prof.export_chrome_trace(str(path))
            import json as _json

            for ev in _json.loads(path.read_text()).get("traceEvents", []):
                name = str(ev.get("name", ""))
                direction = _classify_copy(name)
                if direction is None or direction == "DtoD":
                    continue
                nbytes = int(ev.get("args", {}).get("bytes", 0) or 0)
                key = (name, direction)
                if key not in copies:
                    copies[key] = CopyStat(name=name, count=0, total_bytes=0, direction=direction)
                copies[key].count += 1
                copies[key].total_bytes += nbytes
            written_trace = str(path)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            notes.append(f"chrome trace export failed ({type(exc).__name__}: {exc})")

    # --- recompiles ---------------------------------------------------------------
    counters_delta: Dict[str, Any] = {}
    if cfg.compile and dynamo_counters is not None:
        for k, v in dynamo_counters.items():
            before = counters_before.get(k, {})
            delta = {ik: iv - before.get(ik, 0) for ik, iv in v.items() if iv - before.get(ik, 0)}
            if delta:
                counters_delta[k] = delta

    return ProfileReport(
        device=torch.cuda.get_device_name(device) if is_cuda else "cpu",
        steps=steps,
        compiled=cfg.compile,
        config={"batch_size": cfg.batch_size, "seq_len": cfg.seq_len, "d_model": cfg.d_model},
        top_ops=ops[:top_k],
        copies=sorted(copies.values(), key=lambda c: -c.total_bytes),
        recompile_messages=list(dict.fromkeys(handler.messages)),
        dynamo_counters=counters_delta,
        total_self_time_us=total_self,
        notes=notes,
        trace_path=written_trace,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for ``python -m xjepa.perf.profile_report``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = BenchConfig()
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--seq-len", type=int, default=d.seq_len)
    p.add_argument("--corpus-seqs", type=int, default=d.corpus_seqs)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--model", choices=("synthetic", "xjepa"), default=d.model)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out", type=str, default=None, help="write the markdown report here")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    cfg = BenchConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        corpus_seqs=args.corpus_seqs,
        device=args.device,
        model=args.model,
        compile=args.compile,
    )
    report = run_profile(cfg, steps=args.steps, warmup=args.warmup, top_k=args.top_k)
    md = report.to_markdown()
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"[profile_report] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
