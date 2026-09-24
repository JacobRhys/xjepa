"""Phase 4-6 drivers: grid construction, aggregation and hypothesis verdicts.

The aggregator turns numbers into claims about hypotheses, so its logic is
tested on constructed inputs where the right answer is known. A verdict function
that silently mis-scores would produce a confident, wrong conclusion in the
write-up -- the most expensive kind of bug in this repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.aggregate_results import (
    RunResult,
    by_condition,
    load_results,
    parse_name,
    seed_means,
    seed_variance_check,
    sign_test,
    verdict_h1a,
    verdict_h2,
    verdict_h4,
)
from scripts.run_grid import build_grid


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #


def test_grid_is_23_runs_in_tier_order() -> None:
    grid = build_grid()
    assert len(grid) == 23, "RESEARCH_PLAN sec. 4 specifies 23 runs"
    tiers = [r.tier for r in grid]
    assert tiers == sorted(tiers), "tiers must run in priority order"
    assert sum(t == 1 for t in tiers) == 9, "tier 1 is C1/C2/C3 x 3 seeds"


def test_grid_run_names_are_unique_and_match_trainer_convention() -> None:
    grid = build_grid()
    names = [r.name for r in grid]
    assert len(names) == len(set(names)), "a duplicate name would overwrite a run"
    for run in grid:
        assert run.name == f"{run.run_name}-seed{run.seed}"


def test_grid_variants_do_not_collide_with_their_base_config() -> None:
    """The lambda sweep and 3Di runs reuse base configs; suffixes must separate them."""
    grid = build_grid()
    c4 = {r.name for r in grid if r.config.endswith("c4_mlm_jepa.yaml")}
    assert "c4_mlm_jepa-seed0" in c4
    assert "c4_mlm_jepa_lam0.1-seed0" in c4
    assert "c4_mlm_jepa_lam1.0-seed0" in c4


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "wins,losses,expected",
    [(0, 0, 1.0), (4, 0, 0.125), (8, 0, 2 / 256), (4, 4, 1.0)],
)
def test_sign_test(wins: int, losses: int, expected: float) -> None:
    assert sign_test(wins, losses) == pytest.approx(expected, rel=1e-6)


def test_parse_name() -> None:
    assert parse_name("c3_jepa_frozen-seed2") == ("c3_jepa_frozen", 2)
    assert parse_name("c4_mlm_jepa_lam0.1-seed0") == ("c4_mlm_jepa_lam0.1", 0)
    assert parse_name("baseline_random") == ("baseline_random", 0)


# --------------------------------------------------------------------------- #
# hypothesis verdicts
# --------------------------------------------------------------------------- #


def _runs(cond: str, task_values: dict[str, list[float]], rankme: list[float] | None = None):
    out = []
    n = len(next(iter(task_values.values())))
    for s in range(n):
        out.append(
            RunResult(
                name=f"{cond}-seed{s}",
                condition=cond,
                seed=s,
                metrics={t: v[s] for t, v in task_values.items()},
                rankme=rankme[s] if rankme else None,
            )
        )
    return out


def test_h1a_detects_collapse() -> None:
    groups = {
        "c2_jepa_ema": _runs("c2_jepa_ema", {"ss3": [0.4, 0.4, 0.4]}, rankme=[2.0, 2.1, 1.9]),
        "c3_jepa_frozen": _runs("c3_jepa_frozen", {"ss3": [0.7, 0.7, 0.7]}, rankme=[95.0, 96.0, 94.0]),
    }
    v = verdict_h1a(groups)
    assert v["verdict"] == "SUPPORTED"
    assert v["ratio"] < 0.25


def test_h1a_reports_failed_replication_rather_than_passing_quietly() -> None:
    """If C2 does not collapse, the verdict must say the replication failed."""
    groups = {
        "c2_jepa_ema": _runs("c2_jepa_ema", {"ss3": [0.6] * 3}, rankme=[90.0, 91.0, 89.0]),
        "c3_jepa_frozen": _runs("c3_jepa_frozen", {"ss3": [0.7] * 3}, rankme=[95.0, 96.0, 94.0]),
    }
    v = verdict_h1a(groups)
    assert v["verdict"] == "NOT SUPPORTED"
    assert "replication failed" in v["note"]


def test_h2_requires_the_gap_to_clear_one_sd() -> None:
    """A gap smaller than seed noise must not count as a win."""
    noisy = {
        "c1_mlm": _runs("c1_mlm", {"ss3": [0.70, 0.60, 0.65], "ss8": [0.5, 0.4, 0.45]}),
        # higher mean, but well inside the seed spread
        "c3_jepa_frozen": _runs("c3_jepa_frozen", {"ss3": [0.72, 0.62, 0.67], "ss8": [0.52, 0.42, 0.47]}),
    }
    v = verdict_h2(seed_means(noisy))
    assert v["verdict"] == "NOT SUPPORTED"
    assert all(not c["clears_1sd"] for c in v["comparisons"])

    clean = {
        "c1_mlm": _runs("c1_mlm", {"ss3": [0.60, 0.601, 0.599], "ss8": [0.40, 0.401, 0.399]}),
        "c3_jepa_frozen": _runs("c3_jepa_frozen", {"ss3": [0.75, 0.751, 0.749], "ss8": [0.55, 0.551, 0.549]}),
    }
    v2 = verdict_h2(seed_means(clean))
    assert v2["verdict"] == "SUPPORTED"


def test_h4_null_is_reported_as_a_result_not_a_failure() -> None:
    """C3 ~= C5 is the pre-registered modal outcome and must read as a finding."""
    tasks = {t: [0.5, 0.5] for t in ("ss3", "ss8", "fluorescence", "stability")}
    groups = {
        "c3_jepa_frozen": _runs("c3_jepa_frozen", tasks),
        "c5b_masked_distil": _runs("c5b_masked_distil", tasks),
        "c5c_predictor_nomask": _runs("c5c_predictor_nomask", tasks),
    }
    v = verdict_h4(seed_means(groups))
    assert v["verdict"] == "NOT SUPPORTED"
    assert "not a failure" in v["note"]
    assert "structure supervision" in v["note"]


def test_seed_variance_flags_when_noise_dominates() -> None:
    groups = {
        "c1_mlm": _runs("c1_mlm", {"ss3": [0.50, 0.70, 0.60]}),
        "c3_jepa_frozen": _runs("c3_jepa_frozen", {"ss3": [0.52, 0.72, 0.62]}),
    }
    v = seed_variance_check(groups, seed_means(groups))
    assert v["tasks_where_seed_noise_dominates"] == ["ss3"]
    assert "no condition ranking is supportable" in v["note"]


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


def test_aggregator_runs_end_to_end(tmp_path: Path) -> None:
    from scripts.aggregate_results import main as aggregate_main

    results = tmp_path / "results"
    for cond, acc, rank in (
        ("c1_mlm", 0.60, 180.0),
        ("c2_jepa_ema", 0.35, 2.0),
        ("c3_jepa_frozen", 0.72, 95.0),
    ):
        for seed in range(3):
            d = results / f"{cond}-seed{seed}"
            d.mkdir(parents=True)
            (d / "eval.json").write_text(
                json.dumps(
                    {
                        "label": f"{cond}-seed{seed}",
                        "tasks": {
                            "ss3": {"accuracy": acc + 0.001 * seed},
                            "fold_retrieval": {"top1_accuracy": acc / 2},
                        },
                        "collapse": {"rankme": rank},
                    }
                ),
                encoding="utf-8",
            )

    out = tmp_path / "report"
    assert aggregate_main(["--results", str(results), "--out", str(out)]) == 0

    report = (out / "results.md").read_text(encoding="utf-8")
    assert "# Results" in report
    assert "H1a — SUPPORTED" in report
    assert "Seed variance" in report

    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["hypotheses"]["H1a"]["verdict"] == "SUPPORTED"
    assert payload["conditions"]["c3_jepa_frozen"]["ss3"]["n_seeds"] == 3


def test_load_results_separates_baselines(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for name in ("c1_mlm-seed0", "baseline_random", "baseline_esmif1"):
        d = results / name
        d.mkdir(parents=True)
        (d / "eval.json").write_text(
            json.dumps({"label": name, "tasks": {"ss3": {"accuracy": 0.5}}}), encoding="utf-8"
        )
    loaded = load_results(results)
    conds = by_condition(loaded)
    assert "baseline_random" in conds and "c1_mlm" in conds
