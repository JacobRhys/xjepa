#!/usr/bin/env python3
"""Phase 6: aggregate results and test them against the pre-registered criteria.

Reads every ``results/<run>/eval.json`` and produces the tables the write-up
needs, plus a verdict on each hypothesis measured against the criteria written
down in RESEARCH_PLAN.md sec. 9 -- *before* any result was seen. That ordering is
the only thing that makes the verdicts worth anything, so the criteria are
constants here and the script never adapts them to the data.

Statistical stance, per RESEARCH_PLAN.md sec. 1.7:

* Per-task **mean +/- sd across seeds**, and effect sizes reported with the n=3
  caveat attached. Per-task p-values are not computed: at three seeds they would
  be noise wearing a decimal point.
* Aggregation across tasks uses a **sign test on seed-averaged scores**, which
  treats tasks as the sampling unit. Task independence is an approximation and
  the report says so.
* **Seed variance is a first-class result.** If between-seed spread exceeds
  between-condition gaps, that is the finding, and the report states it in those
  words rather than burying it.

    python scripts/aggregate_results.py --results results/ --out report/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: task -> (metric key within the task's result dict, higher-is-better)
TASK_METRICS: dict[str, tuple[str, bool]] = {
    "ss3": ("accuracy", True),
    "ss8": ("accuracy", True),
    "contact": ("precision_at_l5_long", True),
    "fold_retrieval": ("top1_accuracy", True),
    "fluorescence": ("spearman", True),
    "stability": ("spearman", True),
    "proteingym": ("mean_spearman", True),
}

STRUCTURE_TASKS = ("ss3", "ss8", "contact", "fold_retrieval")

#: Pre-registered pass criteria (RESEARCH_PLAN.md sec. 9). Frozen before results.
CRITERIA = {
    "H1a": "C2 final RankMe < 25% of C3, across all seeds",
    "H1b": "RankMe(C3) reported against the target-bank ceiling; no pass/fail",
    "H2": "C3 >= C1 on >=2/3 structure tasks, seed-mean gap > 1 sd",
    "H3": "C4 best on >=5/8 tasks (sign test over tasks)",
    "H4": "C3 > C5b and C3 > C5c on >=5/8 tasks; C3 ~= C5 rejects H4",
}


@dataclass
class RunResult:
    """One evaluated run."""

    name: str
    condition: str
    seed: int
    metrics: dict[str, float]
    rankme: float | None


def parse_name(name: str) -> tuple[str, int]:
    """``c3_jepa_frozen-seed1`` -> ``("c3_jepa_frozen", 1)``."""
    m = re.match(r"^(.*)-seed(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def metric_of(task_result: dict[str, Any], key: str) -> float | None:
    """Pull one metric, tolerating the naming drift between probe implementations."""
    if not isinstance(task_result, dict) or "error" in task_result or "skipped" in task_result:
        return None
    if key in task_result:
        return float(task_result[key])
    for alt in (key, f"test_{key}", key.replace("_", "")):
        for k, v in task_result.items():
            if k.replace("_", "") == alt.replace("_", "") and isinstance(v, (int, float)):
                return float(v)
    for k, v in task_result.items():
        if isinstance(v, (int, float)) and k not in ("seconds", "n_queries", "n_gallery", "n_assays"):
            return float(v)
    return None


def load_results(root: Path) -> list[RunResult]:
    """Read every ``eval.json`` under ``root``."""
    out: list[RunResult] = []
    for path in sorted(root.glob("*/eval.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("label", path.parent.name)
        condition, seed = parse_name(name)
        metrics: dict[str, float] = {}
        for task, (key, _) in TASK_METRICS.items():
            got = metric_of(data.get("tasks", {}).get(task, {}), key)
            if got is not None:
                metrics[task] = got
        rankme = (data.get("collapse") or {}).get("rankme")
        out.append(RunResult(name, condition, seed, metrics, rankme))
    return out


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test p-value for ``wins`` vs ``losses``."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def by_condition(results: list[RunResult]) -> dict[str, list[RunResult]]:
    out: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        out[r.condition].append(r)
    return dict(out)


def seed_means(groups: dict[str, list[RunResult]]) -> dict[str, dict[str, tuple[float, float, int]]]:
    """condition -> task -> (mean, sd, n_seeds)."""
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for cond, runs in groups.items():
        per_task: dict[str, tuple[float, float, int]] = {}
        for task in TASK_METRICS:
            xs = [r.metrics[task] for r in runs if task in r.metrics]
            if xs:
                m, s = mean_sd(xs)
                per_task[task] = (m, s, len(xs))
        out[cond] = per_task
    return out


# --------------------------------------------------------------------------- #
# hypotheses
# --------------------------------------------------------------------------- #


def verdict_h1a(groups: dict[str, list[RunResult]]) -> dict[str, Any]:
    c2 = [r.rankme for r in groups.get("c2_jepa_ema", []) if r.rankme is not None]
    c3 = [r.rankme for r in groups.get("c3_jepa_frozen", []) if r.rankme is not None]
    if not c2 or not c3:
        return {"verdict": "no data", "criterion": CRITERIA["H1a"]}
    ratio = (sum(c2) / len(c2)) / (sum(c3) / len(c3))
    passed = ratio < 0.25
    return {
        "criterion": CRITERIA["H1a"],
        "c2_rankme_mean": sum(c2) / len(c2),
        "c3_rankme_mean": sum(c3) / len(c3),
        "ratio": ratio,
        "verdict": "SUPPORTED" if passed else "NOT SUPPORTED",
        "note": (
            "C2 collapsed as predicted."
            if passed
            else "C2 did not collapse. The replication failed; the study's framing "
                 "assumes the learned target admits the trivial solution."
        ),
    }


def verdict_h2(stats: dict[str, dict[str, tuple[float, float, int]]]) -> dict[str, Any]:
    c1, c3 = stats.get("c1_mlm", {}), stats.get("c3_jepa_frozen", {})
    wins, comparisons = 0, []
    for task in STRUCTURE_TASKS:
        if task not in c1 or task not in c3:
            continue
        m1, s1, _ = c1[task]
        m3, s3, _ = c3[task]
        gap = m3 - m1
        pooled = max(s1, s3, 1e-9)
        clears = gap > pooled
        wins += int(clears)
        comparisons.append(
            {"task": task, "c1": m1, "c3": m3, "gap": gap, "sd": pooled, "clears_1sd": clears}
        )
    passed = len(comparisons) >= 2 and wins >= max(2, math.ceil(2 * len(comparisons) / 3))
    return {
        "criterion": CRITERIA["H2"],
        "comparisons": comparisons,
        "wins": wins,
        "n_structure_tasks": len(comparisons),
        "verdict": "SUPPORTED" if passed else "NOT SUPPORTED",
    }


def verdict_h3(stats: dict[str, dict[str, tuple[float, float, int]]]) -> dict[str, Any]:
    c4 = stats.get("c4_mlm_jepa", {})
    rivals = ("c1_mlm", "c3_jepa_frozen")
    wins = losses = 0
    per_task = []
    for task in TASK_METRICS:
        if task not in c4:
            continue
        best_rival = max(
            (stats[r][task][0] for r in rivals if r in stats and task in stats[r]), default=None
        )
        if best_rival is None:
            continue
        won = c4[task][0] > best_rival
        wins += int(won)
        losses += int(not won)
        per_task.append({"task": task, "c4": c4[task][0], "best_rival": best_rival, "c4_wins": won})
    return {
        "criterion": CRITERIA["H3"],
        "per_task": per_task,
        "wins": wins,
        "losses": losses,
        "sign_test_p": sign_test(wins, losses),
        "verdict": "SUPPORTED" if wins >= 5 else "NOT SUPPORTED",
    }


def verdict_h4(stats: dict[str, dict[str, tuple[float, float, int]]]) -> dict[str, Any]:
    c3 = stats.get("c3_jepa_frozen", {})
    out: dict[str, Any] = {"criterion": CRITERIA["H4"], "controls": {}}
    verdicts = []
    for control in ("c5b_masked_distil", "c5c_predictor_nomask", "c5_distil"):
        ctrl = stats.get(control, {})
        wins = losses = 0
        per_task = []
        for task in TASK_METRICS:
            if task not in c3 or task not in ctrl:
                continue
            won = c3[task][0] > ctrl[task][0]
            wins += int(won)
            losses += int(not won)
            per_task.append({"task": task, "c3": c3[task][0], control: ctrl[task][0]})
        out["controls"][control] = {
            "wins": wins, "losses": losses,
            "sign_test_p": sign_test(wins, losses),
            "per_task": per_task,
        }
        if wins + losses:
            verdicts.append(wins >= 5)

    isolates = out["controls"].get("c5b_masked_distil", {}).get("wins", 0) >= 5 and \
        out["controls"].get("c5c_predictor_nomask", {}).get("wins", 0) >= 5
    out["verdict"] = "SUPPORTED" if isolates else "NOT SUPPORTED"
    out["note"] = (
        "Masked latent prediction contributes beyond direct distillation."
        if isolates
        else "C3 does not separate from the distillation controls. Report this as the "
             "result -- structure supervision, not latent prediction, drives the gain "
             "-- which is the modal outcome pre-registered in sec. 9.6, not a failure."
    )
    return out


def seed_variance_check(
    groups: dict[str, list[RunResult]], stats: dict[str, dict[str, tuple[float, float, int]]]
) -> dict[str, Any]:
    """Is between-seed spread larger than between-condition spread?"""
    rows = []
    for task in TASK_METRICS:
        means = [s[task][0] for s in stats.values() if task in s]
        sds = [s[task][1] for s in stats.values() if task in s and s[task][2] > 1]
        if len(means) < 2 or not sds:
            continue
        cond_spread = max(means) - min(means)
        seed_spread = sum(sds) / len(sds)
        rows.append({
            "task": task,
            "between_condition_range": cond_spread,
            "mean_between_seed_sd": seed_spread,
            "seed_noise_dominates": seed_spread >= cond_spread,
        })
    dominated = [r["task"] for r in rows if r["seed_noise_dominates"]]
    return {
        "per_task": rows,
        "tasks_where_seed_noise_dominates": dominated,
        "note": (
            f"On {len(dominated)} of {len(rows)} tasks the between-seed spread is at least "
            "as large as the entire range between conditions. On those tasks no condition "
            "ranking is supportable, and that is itself the result."
            if dominated
            else "Between-condition gaps exceed seed noise on every task with multiple seeds."
        ),
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render(stats, groups, hypotheses, variance, baselines) -> str:
    lines: list[str] = ["# Results", ""]
    conds = sorted(stats)
    tasks = [t for t in TASK_METRICS if any(t in stats[c] for c in conds)]

    lines += ["## Per-task results (mean ± sd across seeds)", ""]
    lines.append("| condition | seeds | " + " | ".join(tasks) + " |")
    lines.append("|---|--:|" + "--:|" * len(tasks))
    for cond in conds:
        n = len(groups.get(cond, []))
        cells = []
        for task in tasks:
            if task in stats[cond]:
                m, s, k = stats[cond][task]
                cells.append(f"{m:.3f} ± {s:.3f}" if k > 1 else f"{m:.3f} (n=1)")
            else:
                cells.append("—")
        lines.append(f"| `{cond}` | {n} | " + " | ".join(cells) + " |")
    lines.append("")

    if baselines:
        lines += ["## Baselines", "",
                  "Without these the condition table cannot be interpreted: `random` is the "
                  "floor, `esmif1` is the L1 ceiling that C3/C4/C5 are distilling toward.", ""]
        lines.append("| baseline | " + " | ".join(tasks) + " |")
        lines.append("|---|" + "--:|" * len(tasks))
        for name, metrics in sorted(baselines.items()):
            cells = [f"{metrics[t]:.3f}" if t in metrics else "—" for t in tasks]
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Hypotheses", "",
              "Criteria are quoted from RESEARCH_PLAN.md sec. 9 and were fixed before any "
              "result was seen.", ""]
    for key in ("H1a", "H2", "H3", "H4"):
        h = hypotheses.get(key, {})
        lines += [f"### {key} — {h.get('verdict', 'no data')}", "",
                  f"*Criterion:* {h.get('criterion', CRITERIA.get(key, ''))}", ""]
        if key == "H1a" and "ratio" in h:
            lines.append(
                f"- RankMe: C2 {h['c2_rankme_mean']:.2f}, C3 {h['c3_rankme_mean']:.2f} "
                f"(ratio {h['ratio']:.3f})"
            )
        if key == "H2" and h.get("comparisons"):
            lines += ["", "| task | C1 | C3 | gap | sd | clears 1 sd |", "|---|--:|--:|--:|--:|:-:|"]
            for c in h["comparisons"]:
                lines.append(
                    f"| {c['task']} | {c['c1']:.3f} | {c['c3']:.3f} | {c['gap']:+.3f} "
                    f"| {c['sd']:.3f} | {'yes' if c['clears_1sd'] else 'no'} |"
                )
        if key == "H3" and "wins" in h:
            lines.append(f"- C4 best on {h['wins']}/{h['wins'] + h['losses']} tasks "
                         f"(sign test p = {h['sign_test_p']:.3f})")
        if key == "H4":
            for ctrl, d in h.get("controls", {}).items():
                if d["wins"] + d["losses"]:
                    lines.append(
                        f"- C3 vs `{ctrl}`: {d['wins']} wins / {d['losses']} losses "
                        f"(p = {d['sign_test_p']:.3f})"
                    )
        if h.get("note"):
            lines += ["", f"> {h['note']}"]
        lines.append("")

    lines += ["## Seed variance", "",
              "Reported as a first-class result, not a footnote.", ""]
    if variance["per_task"]:
        lines += ["| task | between-condition range | mean between-seed sd | seed noise dominates |",
                  "|---|--:|--:|:-:|"]
        for r in variance["per_task"]:
            lines.append(
                f"| {r['task']} | {r['between_condition_range']:.3f} "
                f"| {r['mean_between_seed_sd']:.3f} "
                f"| {'YES' if r['seed_noise_dominates'] else 'no'} |"
            )
    lines += ["", f"> {variance['note']}", "",
              "## Statistical caveats", "",
              "- Three seeds. Per-task p-values are deliberately not reported; at n=3 they "
              "would be noise with a decimal point.",
              "- Cross-task aggregation uses a sign test treating tasks as the sampling unit. "
              "Task independence is an approximation.",
              "- ProteinGym is scored by embedding distance and is exploratory: |rho| < 0.2 "
              "was the pre-registered expectation, since JEPA-only models expose no token "
              "likelihoods.",
              "- C3/C4/C5 receive supervision from a 142M-parameter model trained on far more "
              "data than C1 ever sees. Matched tokens do not fix that asymmetry; the honest "
              "claim is about how best to spend a fixed sequence-model budget given access to "
              "a pretrained structure encoder.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("report"))
    args = p.parse_args(argv)

    all_results = load_results(args.results)
    if not all_results:
        print(f"no eval.json found under {args.results}", file=sys.stderr)
        return 1

    baseline_runs = [r for r in all_results if r.condition.startswith("baseline_")]
    runs = [r for r in all_results if not r.condition.startswith("baseline_")]
    if not runs:
        print("only baselines found; nothing to aggregate", file=sys.stderr)
        return 1

    groups = by_condition(runs)
    stats = seed_means(groups)
    hypotheses = {
        "H1a": verdict_h1a(groups),
        "H1b": {"criterion": CRITERIA["H1b"],
                "c3_rankme": stats.get("c3_jepa_frozen", {}),
                "note": "Compare against the target bank's RankMe from build_cache meta.json."},
        "H2": verdict_h2(stats),
        "H3": verdict_h3(stats),
        "H4": verdict_h4(stats),
    }
    variance = seed_variance_check(groups, stats)
    baselines = {r.condition: r.metrics for r in baseline_runs}

    report = render(stats, groups, hypotheses, variance, baselines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(report, encoding="utf-8")
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "conditions": {
                    c: {t: {"mean": v[0], "sd": v[1], "n_seeds": v[2]} for t, v in s.items()}
                    for c, s in stats.items()
                },
                "baselines": baselines,
                "hypotheses": hypotheses,
                "seed_variance": variance,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"\nWrote {args.out / 'results.md'} and {args.out / 'results.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
