"""Performance harness: benchmark and profiler report.

This package is the only place in the repo permitted to call
``torch.cuda.synchronize()`` (docs/CONTRACTS.md, hard rule 2) -- measurement
needs it, training must never pay for it.
"""

from __future__ import annotations

from .bench import (
    PEAK_BF16_TFLOPS,
    REFERENCE_PARAMS,
    BenchConfig,
    BenchModel,
    BenchResult,
    SyntheticCorpus,
    detect_sdpa_backend,
    peak_bf16_tflops,
    run_benchmark,
    transformer_flops_per_step,
)
from .profile_report import CopyStat, OpStat, ProfileReport, run_profile

__all__ = [
    "BenchConfig",
    "BenchResult",
    "BenchModel",
    "SyntheticCorpus",
    "PEAK_BF16_TFLOPS",
    "REFERENCE_PARAMS",
    "peak_bf16_tflops",
    "transformer_flops_per_step",
    "detect_sdpa_backend",
    "run_benchmark",
    "OpStat",
    "CopyStat",
    "ProfileReport",
    "run_profile",
]
