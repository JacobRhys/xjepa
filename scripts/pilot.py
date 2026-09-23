#!/usr/bin/env python3
"""Pilot: measure throughput, then recompute the study budget from what it measured.

RESEARCH_PLAN.md sec. 6.1 makes this a gate -- the grid does not launch until the
modelled numbers have been replaced by measured ones. The plan's cost table
assumes 25% MFU; ``docs/PERF.md`` argues the realistic range is 18-25%, and the
difference is about $1 of a ~£6 study, so it is worth 20 minutes to find out.

Two open decisions are settled here, both by measurement rather than argument
(RESEARCH_PLAN.md sec. 3.4):

**Bucket policy -- does removing the attention mask reach the flash backend?**
PyTorch's flash SDPA path accepts only ``is_causal`` or no mask at all, so any
key-padding mask drops us to cutlassF. The ``crop`` policy crops every sequence
to exactly its bucket length, leaving zero padding and therefore no mask;
``hybrid`` pads up when bucket occupancy is >= 0.85 and reaches only 95.7%
padding efficiency, but keeps 7.6 percentage points more of each pass's
residues. Modelled gain is ~1.4x; this measures it.

**Head count -- is head_dim 16 leaving the tensor cores half empty?**
ESM-2 t6 uses 20 heads, giving ``head_dim = 320/20 = 16``. Eight heads gives 40.
Attention parameters are ``d_model x d_model`` regardless of head count, so the
parameter budget is *identical* and the experimental control is untouched as
long as every condition shares the choice. The only cost of moving is fidelity
to the reference config.

Run::

    python scripts/pilot.py                      # full 2x2 sweep on GPU
    python scripts/pilot.py --smoke              # tiny shapes, runs on CPU
    python scripts/pilot.py --out runs/pilot     # save JSON + markdown

The exit code is non-zero if a contract invariant fails (host-to-device bytes in
the hot loop, implicit syncs, or SDPA falling back to the math kernel), so this
can gate a launch script.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

# Runnable as `python scripts/pilot.py` from the repo root without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xjepa.perf.bench import BenchConfig, BenchResult, run_benchmark

# --------------------------------------------------------------------------- #
# study constants (RESEARCH_PLAN.md sec. 4 and sec. 6)
# --------------------------------------------------------------------------- #

TOKENS_PER_RUN = 1_000_000_000
N_RUNS = 23
USD_PER_GBP = 1.26
DEFAULT_RATE_USD_PER_HOUR = 0.34  # RunPod community RTX 4090, verify at booking
BUDGET_GBP = 10.0

#: GPU hours the plan's cost table spends outside the pretraining grid:
#: calibration, the ESM-IF1 target cache, probe feature extraction, contact probe.
FIXED_GPU_HOURS = 2.0 + 3.5 + 1.6 + 1.5
#: Network volume, 20 GB for a month.
STORAGE_USD = 1.40
#: Plan sec. 6 contingency line.
CONTINGENCY_GPU_HOURS = 2.0


@dataclass(frozen=True)
class Cell:
    """One cell of the sweep."""

    policy: str  # "hybrid" | "crop"
    n_heads: int

    @property
    def label(self) -> str:
        return f"{self.policy}/{self.n_heads}h"

    @property
    def attn_mask(self) -> bool:
        """`crop` leaves zero padding, so it passes no mask and can reach flash."""
        return self.policy == "hybrid"

    @property
    def head_dim_note(self) -> str:
        return f"head_dim={320 // self.n_heads}"


@dataclass
class Costing:
    """The budget implied by one measured throughput."""

    label: str
    tokens_per_sec: float
    mfu: float | None
    minutes_per_run: float
    grid_gpu_hours: float
    total_gpu_hours: float
    total_usd: float
    total_gbp: float
    headroom_gbp: float
    within_budget: bool


def cost_from_throughput(
    label: str,
    tokens_per_sec: float,
    mfu: float | None,
    rate: float,
    contingency: bool = True,
) -> Costing:
    """Turn a measured tokens/sec into the whole study's cost.

    Args:
        label: Name of the cell this came from.
        tokens_per_sec: Measured training throughput.
        mfu: Measured model FLOP utilisation, if the card's peak is known.
        rate: USD per GPU hour.
        contingency: Include the plan's 2-hour contingency line.

    Returns:
        The implied :class:`Costing`.
    """
    seconds_per_run = TOKENS_PER_RUN / max(tokens_per_sec, 1e-9)
    grid_hours = seconds_per_run * N_RUNS / 3600.0
    total_hours = grid_hours + FIXED_GPU_HOURS + (CONTINGENCY_GPU_HOURS if contingency else 0.0)
    usd = total_hours * rate + STORAGE_USD
    gbp = usd / USD_PER_GBP
    return Costing(
        label=label,
        tokens_per_sec=tokens_per_sec,
        mfu=mfu,
        minutes_per_run=seconds_per_run / 60.0,
        grid_gpu_hours=grid_hours,
        total_gpu_hours=total_hours,
        total_usd=usd,
        total_gbp=gbp,
        headroom_gbp=BUDGET_GBP - gbp,
        within_budget=gbp <= BUDGET_GBP,
    )


def build_cells(policies: Iterable[str], heads: Iterable[int]) -> list[Cell]:
    return [Cell(policy=p, n_heads=h) for p in policies for h in heads]


def bench_config_for(cell: Cell, base: BenchConfig) -> BenchConfig:
    """Specialise the base benchmark config for one sweep cell."""
    return replace(base, n_heads=cell.n_heads, attn_mask=cell.attn_mask)


# --------------------------------------------------------------------------- #
# invariant checks (docs/PERF.md reviewer checklist)
# --------------------------------------------------------------------------- #


def check_invariants(result: BenchResult) -> list[str]:
    """Return a list of violated contract invariants (empty means clean).

    These are the claims the whole design rests on, so the pilot asserts them
    rather than trusting them: no host-to-device traffic in the hot loop, no
    implicit device syncs, and attention not silently on the math kernel.
    """
    problems: list[str] = []

    h2d = result.h2d_bytes_per_step
    if h2d is not None and h2d > 0:
        problems.append(
            f"host->device traffic in the hot loop: {h2d:,.0f} B/step "
            "(the corpus is meant to be resident; something is being transferred)"
        )

    syncs = result.implicit_syncs
    if syncs:
        problems.append(
            f"{syncs} implicit device sync(s) in the hot loop -- "
            f"first: {result.sync_messages[0] if result.sync_messages else 'n/a'}"
        )

    if result.sdpa_backend == "math":
        problems.append(
            "SDPA fell back to the math kernel -- check the attention mask dtype "
            "matches the autocast compute dtype"
        )

    return problems


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _fmt_mfu(mfu: float | None) -> str:
    return f"{mfu * 100:.1f}%" if mfu is not None else "n/a"


def render_report(
    results: dict[str, BenchResult],
    costings: dict[str, Costing],
    baseline: str,
    rate: float,
) -> str:
    """Render the decision table, budget recomputation and recommendation.

    On a non-CUDA device the timings are meaningless -- SDPA never dispatches,
    so the flash-vs-cutlassF question the sweep exists to answer cannot even be
    asked. The report then shows what it measured but withholds every decision
    and cost figure, rather than dressing CPU noise as a recommendation.
    """
    lines: list[str] = []
    add = lines.append

    first = next(iter(results.values()))
    valid = first.cuda_available and not first.sdpa_backend.startswith("cpu")

    add("# Pilot report")
    add("")
    if not valid:
        add("> **NOT A RESULT -- no CUDA device.**")
        add("> These timings are CPU noise. SDPA never dispatches on CPU, so the")
        add("> flash-vs-cutlassF question this sweep exists to answer was not asked,")
        add("> and the cost model would be nonsense. Decisions and budget are")
        add("> withheld. Run this on the rented GPU. This mode only proves the")
        add("> script itself works end to end.")
        add("")
    add(f"- Device: **{first.device_name}**  ·  torch {first.torch_version}")
    add(f"- Rate assumed: ${rate:.2f}/GPU-hour  ·  budget £{BUDGET_GBP:.0f}")
    add(f"- Params used for FLOPs: {first.flops_params_used:,} ({first.params_basis})")
    add("")

    add("## Measured throughput")
    add("")
    add("| cell | | steps/s | tokens/s | MFU | SDPA backend | H2D B/step | syncs |")
    add("|---|---|--:|--:|--:|---|--:|--:|")
    for label, r in results.items():
        cell_note = next(c.head_dim_note for c in ALL_CELLS if c.label == label)
        h2d = "n/a" if r.h2d_bytes_per_step is None else f"{r.h2d_bytes_per_step:,.0f}"
        syncs = "n/a" if r.implicit_syncs is None else str(r.implicit_syncs)
        add(
            f"| `{label}` | {cell_note} | {r.steps_per_sec:.2f} | {r.tokens_per_sec:,.0f} "
            f"| {_fmt_mfu(r.mfu)} | {r.sdpa_backend} | {h2d} | {syncs} |"
        )
    add("")

    if not valid:
        add("_Speedup table, budget recomputation and decisions withheld: "
            "no CUDA device._")
        return "\n".join(lines)

    base_r = results[baseline]
    add(f"## Speedup vs `{baseline}` (the plan's current configuration)")
    add("")
    add("| cell | speedup | minutes/run | grid GPU-h | study total | headroom |")
    add("|---|--:|--:|--:|--:|--:|")
    for label, c in costings.items():
        speed = results[label].tokens_per_sec / max(base_r.tokens_per_sec, 1e-9)
        flag = "" if c.within_budget else "  **OVER**"
        add(
            f"| `{label}` | {speed:.2f}x | {c.minutes_per_run:.1f} | {c.grid_gpu_hours:.1f} "
            f"| £{c.total_gbp:.2f}{flag} | £{c.headroom_gbp:.2f} |"
        )
    add("")
    add(
        "Study total covers the whole plan sec. 6 table: the 23-run grid at the measured "
        f"rate, plus {FIXED_GPU_HOURS:.1f} fixed GPU-hours (calibration, ESM-IF1 cache, "
        f"feature extraction, contact probe), {CONTINGENCY_GPU_HOURS:.0f} h contingency "
        f"and ${STORAGE_USD:.2f} storage."
    )
    add("")

    add("## Decisions")
    add("")
    add(_recommend(results, costings, baseline))
    return "\n".join(lines)


def _recommend(
    results: dict[str, BenchResult],
    costings: dict[str, Costing],
    baseline: str,
) -> str:
    """Turn the measurements into the two decisions the plan is waiting on."""
    out: list[str] = []
    base_tps = results[baseline].tokens_per_sec

    def speed(label: str) -> float:
        return results[label].tokens_per_sec / max(base_tps, 1e-9)

    # --- decision 1: bucket policy ---------------------------------------- #
    crop_cells = [lbl for lbl in results if lbl.startswith("crop/")]
    if crop_cells:
        best_crop = max(crop_cells, key=speed)
        gain = speed(best_crop)
        backend = results[best_crop].sdpa_backend
        reached_flash = "flash" in backend.lower()
        out.append("### Bucket policy")
        out.append("")
        if not reached_flash:
            out.append(
                f"- Removing the mask did **not** reach the flash backend (`{backend}`), "
                f"and gained {gain:.2f}x. The 1.4x argument does not hold on this card "
                "or this torch build -- **keep `hybrid`** and keep the 7.6 extra "
                "percentage points of residues per pass."
            )
        elif gain >= 1.2:
            out.append(
                f"- `crop` reaches **{backend}** and is **{gain:.2f}x** faster. "
                "**Take it.** Cropping every sequence to exactly its bucket length "
                "raises cropped residues per pass from 14.2% to 22.6%; since the "
                "budget is counted in tokens and crop windows are redrawn each pass, "
                "that costs coverage rate, not information."
            )
            out.append(
                "- Decide the short tail explicitly: sequences under 128 residues have "
                "no bucket to crop into. Either raise the data-build length floor to "
                "128 (clean, but drops a distinct structural population) or keep one "
                "masked bucket for them (keeps composition, keeps cutlassF for those "
                "batches). Whichever you pick must be identical across all conditions."
            )
        else:
            out.append(
                f"- `crop` reaches {backend} but gains only {gain:.2f}x. Below the 1.2x "
                "bar, **keep `hybrid`** -- churning tested code is not worth this."
            )
        out.append("")

    # --- decision 2: head count ------------------------------------------- #
    policies = {lbl.split("/")[0] for lbl in results}
    per_policy: list[str] = []
    for pol in sorted(policies):
        h20 = f"{pol}/20h"
        h8 = f"{pol}/8h"
        if h20 in results and h8 in results:
            per_policy.append(f"{pol}: {speed(h8) / max(speed(h20), 1e-9):.2f}x")
    if per_policy:
        gains = [float(p.split(": ")[1].rstrip("x")) for p in per_policy]
        best = max(gains)
        out.append("### Head count")
        out.append("")
        out.append(f"- 8 heads (head_dim 40) vs 20 heads (head_dim 16): {', '.join(per_policy)}.")
        if best >= 1.15:
            out.append(
                f"- **Take 8 heads.** At {best:.2f}x it is worth the deviation: attention "
                "parameters are `d_model x d_model` regardless of head count, so the "
                "parameter budget is unchanged and the control across conditions holds. "
                "Note the deviation from ESM-2 t6 in the methods."
            )
        else:
            out.append(
                f"- **Keep 20 heads.** At {best:.2f}x the gain does not pay for losing "
                "literal ESM-2 t6 fidelity, which is one of the few things anchoring "
                "this study to a known reference."
            )
        out.append("")

    # --- decision 3: does the budget still hold? --------------------------- #
    best_label = max(costings, key=lambda k: results[k].tokens_per_sec)
    best = costings[best_label]
    worst = costings[baseline]
    out.append("### Budget")
    out.append("")
    out.append(
        f"- Best cell `{best_label}`: **£{best.total_gbp:.2f}** of £{BUDGET_GBP:.0f}, "
        f"£{best.headroom_gbp:.2f} headroom. Baseline `{baseline}`: £{worst.total_gbp:.2f}."
    )
    if not worst.within_budget:
        out.append(
            "- The baseline configuration does **not** fit the budget. Either take the "
            "faster cell, or cut the grid to the Tier-1 conditions (C1/C2/C3, 3 seeds = "
            "9 runs) as RESEARCH_PLAN.md sec. 4 provides for."
        )
    elif best.headroom_gbp >= 3.0:
        out.append(
            "- Headroom is comfortable. Spend it in the plan's order: third seed on "
            "C5/C5b/C5c first (they are the H4 mechanism controls and the modal outcome "
            "is a null, which needs a spread to report), then the T-masked target "
            "ablation, then 2B-token runs on C1/C2/C3."
        )
    else:
        out.append("- Headroom is thin. Hold the contingency; do not add seeds yet.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

ALL_CELLS: list[Cell] = []


def main(argv: list[str] | None = None) -> int:
    global ALL_CELLS

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=50, help="measured steps per cell")
    p.add_argument("--warmup", type=int, default=10, help="discarded warmup steps")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--rate", type=float, default=DEFAULT_RATE_USD_PER_HOUR,
                   help="USD per GPU hour (verify at booking; prices move)")
    p.add_argument("--heads", type=int, nargs="+", default=[20, 8])
    p.add_argument("--policies", nargs="+", default=["hybrid", "crop"], choices=["hybrid", "crop"])
    p.add_argument("--compile", action="store_true", help="measure with torch.compile (slow to start)")
    p.add_argument("--model", default="xjepa", choices=["xjepa", "synthetic"])
    p.add_argument("--out", type=Path, default=None, help="directory for pilot.json / pilot.md")
    p.add_argument("--smoke", action="store_true",
                   help="tiny shapes so the script runs end to end on CPU")
    args = p.parse_args(argv)

    if args.smoke:
        args.steps, args.warmup = 3, 1
        args.batch_size, args.seq_len = 4, 64
        args.compile = False

    if not torch.cuda.is_available() and not args.smoke:
        print(
            "No CUDA device. The timing numbers this script exists to produce are "
            "meaningless on CPU -- run it on the rented GPU, or pass --smoke to "
            "check the script itself works.",
            file=sys.stderr,
        )
        return 2

    base = BenchConfig(
        steps=args.steps,
        warmup=args.warmup,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        compile=args.compile,
        model=args.model,
        params_basis="encoder",  # pin N=7,408,960 so cells stay comparable
    )

    ALL_CELLS = build_cells(args.policies, args.heads)
    baseline = "hybrid/20h" if any(c.label == "hybrid/20h" for c in ALL_CELLS) else ALL_CELLS[0].label

    results: dict[str, BenchResult] = {}
    problems: dict[str, list[str]] = {}
    for cell in ALL_CELLS:
        print(f"[pilot] {cell.label} ({cell.head_dim_note}, "
              f"{'masked' if cell.attn_mask else 'no mask'}) ...", file=sys.stderr)
        res = run_benchmark(bench_config_for(cell, base), verbose=False)
        results[cell.label] = res
        found = check_invariants(res)
        if found:
            problems[cell.label] = found

    costings = {
        lbl: cost_from_throughput(lbl, r.tokens_per_sec, r.mfu, args.rate)
        for lbl, r in results.items()
    }

    report = render_report(results, costings, baseline, args.rate)
    print(report)

    if problems:
        print("\n## Invariant violations\n")
        for lbl, found in problems.items():
            for msg in found:
                print(f"- `{lbl}`: {msg}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pilot.md").write_text(report, encoding="utf-8")
        payload: dict[str, Any] = {
            "results": {k: v.to_dict() for k, v in results.items()},
            "costings": {k: vars(v) for k, v in costings.items()},
            "problems": problems,
            "rate_usd_per_hour": args.rate,
            "baseline": baseline,
        }
        (args.out / "pilot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out / 'pilot.md'} and {args.out / 'pilot.json'}", file=sys.stderr)

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
