#!/usr/bin/env python3
"""Phase 4: run the condition grid in priority tiers, under a hard spend cap.

The grid is 23 runs (RESEARCH_PLAN.md sec. 4) organised into tiers so that a
budget or time overrun degrades gracefully instead of leaving a half-finished
sweep with no coherent story:

* **Tier 1** -- C1, C2, C3 x 3 seeds. The minimum viable study: answers H1a
  (does the EMA target collapse?) and H2 (does the frozen target beat MLM on
  structure tasks?).
* **Tier 2** -- C4, C5b, C5c x 2 seeds. H3 and the H4 mechanism controls.
* **Tier 3** -- C5, the lambda sweep, the 3Di target variant, span masking.

Between tiers the launcher stops and prints what it found, because there are
real decisions there. If C2 did not collapse, the replication failed and the
entire framing needs rethinking -- and tier 1 is the cheapest possible place to
discover that.

Every run is capped by ``--max-minutes`` (the trainer's own wall-clock stop), and
the launcher tracks cumulative GPU spend against ``--budget-gbp``, refusing to
start a run it cannot afford. Checkpoints are written every 1000 steps, so a
preempted spot instance loses at most that.

    python scripts/run_grid.py --corpus data/corpus --out runs/ --tier 1
    python scripts/run_grid.py --corpus data/corpus --out runs/ --tier 2 --tier 3
    python scripts/run_grid.py --corpus data/corpus --out runs/ --dry-run

Re-running skips runs that already finished, so this is the resume path after a
preemption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

USD_PER_GBP = 1.26


@dataclass(frozen=True)
class Run:
    """One (condition, seed) cell of the grid."""

    config: str
    seed: int
    tier: int
    overrides: tuple[tuple[str, str], ...] = field(default=())
    suffix: str = ""

    @property
    def run_name(self) -> str:
        """The config `name` field, which also names the run directory."""
        return f"{Path(self.config).stem}{self.suffix}"

    @property
    def name(self) -> str:
        """Directory name, matching the trainer's `{name}-seed{seed}` convention."""
        return f"{self.run_name}-seed{self.seed}"


def build_grid() -> list[Run]:
    """The 23 runs of RESEARCH_PLAN.md sec. 4, in execution order."""
    runs: list[Run] = []

    # Tier 1 -- minimum viable study.
    for cfg in ("c1_mlm", "c2_jepa_ema", "c3_jepa_frozen"):
        for seed in (0, 1, 2):
            runs.append(Run(f"configs/{cfg}.yaml", seed, tier=1))

    # Tier 2 -- complementarity and the mechanism controls. C4 carries three
    # seeds because H3 is tested against C1 and C3, both of which have three;
    # the two controls carry two, per RESEARCH_PLAN.md sec. 4.
    for seed in (0, 1, 2):
        runs.append(Run("configs/c4_mlm_jepa.yaml", seed, tier=2))
    for cfg in ("c5b_masked_distil", "c5c_predictor_nomask"):
        for seed in (0, 1):
            runs.append(Run(f"configs/{cfg}.yaml", seed, tier=2))

    # Tier 3 -- joint control, lambda sweep, robustness, masking ablation.
    for seed in (0, 1):
        runs.append(Run("configs/c5_distil.yaml", seed, tier=3))
    for lam in ("0.1", "1.0"):
        runs.append(
            Run(
                "configs/c4_mlm_jepa.yaml", 0, tier=3,
                overrides=(("objective.lambda_jepa", lam),),
                suffix=f"_lam{lam}",
            )
        )
    for seed in (0, 1):
        runs.append(
            Run(
                "configs/c3_jepa_frozen.yaml", seed, tier=3,
                overrides=(("corpus_path", "data/corpus_3di"),),
                suffix="_3di",
            )
        )
    runs.append(
        Run(
            "configs/c3_jepa_frozen.yaml", 0, tier=3,
            overrides=(("objective.masking", "span"),),
            suffix="_span",
        )
    )
    return runs


def is_complete(run_dir: Path) -> bool:
    """A run counts as done when the trainer wrote its summary."""
    return (run_dir / "summary.json").exists()


def collapse_snapshot(run_dir: Path) -> dict[str, float] | None:
    """Read the final collapse metrics out of a finished run's CSV.

    Returns ``None`` when the run has no metrics, rather than raising: the
    between-tier report is diagnostic, and a missing column should not abort
    a grid that is otherwise fine.
    """
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return None
    try:
        import csv as _csv

        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            return None
        last = rows[-1]
        out: dict[str, float] = {}
        for key in ("rankme", "dim_std", "offdiag_cov_mass", "loss"):
            for col in last:
                if col == key or col.endswith(f"_{key}"):
                    try:
                        out[key] = float(last[col])
                    except (TypeError, ValueError):
                        pass
                    break
        return out or None
    except Exception:  # noqa: BLE001 - diagnostics must never break the grid
        return None


def tier_report(runs: list[Run], out_root: Path, tier: int) -> str:
    """Summarise a finished tier, flagging what should be checked before spending more."""
    lines = [f"\n{'=' * 68}", f"Tier {tier} complete", "=" * 68]
    ranks: dict[str, list[float]] = {}
    for run in runs:
        if run.tier != tier:
            continue
        snap = collapse_snapshot(out_root / run.name)
        cond = Path(run.config).stem
        if snap and "rankme" in snap:
            ranks.setdefault(cond, []).append(snap["rankme"])
        shown = (
            "  ".join(f"{k}={v:.4g}" for k, v in snap.items()) if snap else "no metrics found"
        )
        lines.append(f"  {run.name:<34} {shown}")

    if tier == 1 and "c2_jepa_ema" in ranks and "c3_jepa_frozen" in ranks:
        c2 = sum(ranks["c2_jepa_ema"]) / len(ranks["c2_jepa_ema"])
        c3 = sum(ranks["c3_jepa_frozen"]) / len(ranks["c3_jepa_frozen"])
        ratio = c2 / c3 if c3 else float("inf")
        lines += [
            "",
            f"  H1a check: mean RankMe  C2={c2:.2f}  C3={c3:.2f}  (ratio {ratio:.3f})",
        ]
        if ratio < 0.25:
            lines.append("  -> C2 collapsed as predicted. Replication holds; continue.")
        else:
            lines += [
                "  -> C2 did NOT collapse.",
                "     This is the replication failing, not a bug to route around. The",
                "     study's framing assumes the learned target admits the trivial",
                "     solution. Stop and work out why before spending tier 2.",
            ]
    lines.append("=" * 68)
    return "\n".join(lines)


def launch(
    run: Run,
    out_root: Path,
    corpus: str,
    max_minutes: float,
    device: str,
    extra: list[str],
    dry_run: bool,
) -> tuple[bool, float]:
    """Run one cell. Returns ``(ok, elapsed_hours)``."""
    cmd = [
        sys.executable, "-m", "xjepa.train.trainer",
        "--config", run.config,
        "--seed", str(run.seed),
        # The trainer appends `{name}-seed{seed}` itself, so it takes the ROOT.
        "--out-dir", str(out_root),
        "--max-minutes", str(max_minutes),
        "--device", device,
        "--set", f"name={run.run_name}",
    ]
    for key, value in run.overrides:
        cmd += ["--set", f"{key}={value}"]
    cmd += extra

    if dry_run:
        print("  " + " ".join(cmd))
        return True, 0.0

    t0 = time.time()
    proc = subprocess.run(cmd)
    hours = (time.time() - t0) / 3600.0
    if proc.returncode != 0:
        print(f"[grid] {run.name} FAILED (exit {proc.returncode})", file=sys.stderr)
        return False, hours
    return True, hours


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, required=True, help="root directory for run dirs")
    p.add_argument("--corpus", default="data/corpus", help="corpus_path override for every run")
    p.add_argument("--tier", type=int, action="append", choices=[1, 2, 3],
                   help="tiers to run; repeatable. Default: tier 1 only.")
    p.add_argument("--only", nargs="*", default=None, help="run names to run, ignoring tiers")
    p.add_argument("--max-minutes", type=float, default=35.0, help="hard wall-clock cap per run")
    p.add_argument("--rate", type=float, default=0.34, help="USD per GPU hour")
    p.add_argument("--budget-gbp", type=float, default=10.0,
                   help="refuse to start a run once cumulative spend passes this")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="print commands and exit")
    p.add_argument("--force", action="store_true", help="re-run cells that already finished")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument("extra", nargs="*", help="extra args forwarded to the trainer")
    args = p.parse_args(argv)

    tiers = sorted(set(args.tier or [1]))
    grid = build_grid()
    selected = (
        [r for r in grid if r.name in set(args.only)]
        if args.only
        else [r for r in grid if r.tier in tiers]
    )
    if not selected:
        print("no runs selected", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    extra = list(args.extra)
    if args.corpus:
        extra += ["--set", f"corpus_path={args.corpus}"]

    est_hours = len(selected) * args.max_minutes / 60.0
    est_gbp = est_hours * args.rate / USD_PER_GBP
    print(
        f"[grid] {len(selected)} runs across tier(s) {tiers}\n"
        f"[grid] worst case {est_hours:.1f} GPU-h = £{est_gbp:.2f} at ${args.rate:.2f}/h "
        f"(every run hitting the {args.max_minutes:.0f} min cap)\n"
        f"[grid] budget £{args.budget_gbp:.2f}",
        file=sys.stderr,
    )
    if est_gbp > args.budget_gbp:
        print(
            f"[grid] WARNING: worst case exceeds the budget. Runs will be refused once "
            f"cumulative spend passes £{args.budget_gbp:.2f}; tier 1 is ordered first.",
            file=sys.stderr,
        )

    spent_hours = 0.0
    done: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    ledger = args.out / "grid_ledger.json"

    for tier in tiers if not args.only else [0]:
        tier_runs = [r for r in selected if args.only or r.tier == tier]
        for run in tier_runs:
            run_dir = args.out / run.name
            if is_complete(run_dir) and not args.force:
                print(f"[grid] skip {run.name} (already complete)", file=sys.stderr)
                skipped.append(run.name)
                continue

            spent_gbp = spent_hours * args.rate / USD_PER_GBP
            next_gbp = spent_gbp + (args.max_minutes / 60.0) * args.rate / USD_PER_GBP
            if not args.dry_run and next_gbp > args.budget_gbp:
                print(
                    f"[grid] STOPPING before {run.name}: it could take spend to "
                    f"£{next_gbp:.2f}, past the £{args.budget_gbp:.2f} budget.",
                    file=sys.stderr,
                )
                break

            print(f"\n[grid] === {run.name} (tier {run.tier}) ===", file=sys.stderr)
            ok, hours = launch(
                run, args.out, args.corpus, args.max_minutes, args.device, extra, args.dry_run
            )
            spent_hours += hours
            (done if ok else failed).append(run.name)
            if not ok and not args.continue_on_failure:
                print("[grid] stopping on failure (--continue-on-failure to override)",
                      file=sys.stderr)
                break

            ledger.write_text(
                json.dumps(
                    {
                        "done": done, "failed": failed, "skipped": skipped,
                        "gpu_hours": round(spent_hours, 3),
                        "spend_gbp": round(spent_hours * args.rate / USD_PER_GBP, 2),
                        "rate_usd_per_hour": args.rate,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        if not args.dry_run and not args.only:
            print(tier_report(selected, args.out, tier), file=sys.stderr)

    spent_gbp = spent_hours * args.rate / USD_PER_GBP
    print(
        f"\n[grid] {len(done)} ok, {len(failed)} failed, {len(skipped)} skipped\n"
        f"[grid] {spent_hours:.2f} GPU-h = £{spent_gbp:.2f} of £{args.budget_gbp:.2f}",
        file=sys.stderr,
    )
    if failed:
        print(f"[grid] failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
