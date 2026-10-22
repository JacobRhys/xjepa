#!/usr/bin/env python3
"""Check a methods write-up against the code that produced the results.

Every factual number in the methods section has a source of truth in this repo:
the configs, the core grid, the model definitions, the probe grid, the eval data
manifests and the leakage-filter report. This script derives each number from
that source and checks the write-up states it, then scans for figures that were
true of an earlier plan and are not true of the study as run.

    python scripts/check_methods.py paper/methods.md

It is deliberately not part of the pytest suite: wording in a draft must never
block a pod from starting a phase.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Figures that described an earlier version of the plan. Each is wrong for the
#: study as run, so finding one in the write-up means a stale sentence.
STALE = {
    r"15,?300 steps": "v2 step budget (1B tokens); v3 runs 7,630 steps",
    r"PCA-128": "the bank is PCA-256; 128 was rejected at gate 2",
    r"\b15 ?% (of positions|masking|mask)": "mask rate is 20%, not ESM-2's 15%",
    r"1(\.0)? ?B tokens (per|each|a) run": "v3 runs ~0.5B tokens; say 1B only as the original plan",
    r"\b23 runs\b(?! (were )?planned)": "11 runs were trained",
    r"20 (attention )?heads(?! *\))(?!.*rather)": "the encoder uses 8 heads",
    r"\b8M[- ]param": "the encoder is 7.41M parameters; '8M' is ESM-2's round-up",
}


def fmt(n: int) -> str:
    return f"{n:,}"


def facts() -> list[tuple[str, list[str]]]:
    """``(claim, accepted spellings)``; the write-up must contain one spelling."""
    import torch  # noqa: F401  (imported by the model modules anyway)

    from scripts.run_grid import TOKENS_PER_STEP, core_grid
    from xjepa.eval.probes import LINEAR_GRID, MLP_GRID
    from xjepa.model.encoder import Encoder, EncoderConfig
    from xjepa.model.heads import PredictorConfig
    from xjepa.train.trainer import load_config

    cfg = load_config(ROOT / "configs" / "c3_jepa_frozen.yaml")
    out: list[tuple[str, list[str]]] = []
    add = lambda claim, *spellings: out.append((claim, list(spellings)))  # noqa: E731

    # --- training configuration ------------------------------------------- #
    add("mask rate", f"{cfg.mask_rate * 100:.0f}%")
    add("layers", f"{cfg.n_layers} layers")
    add("model width", f"width {cfg.d_model}", f"d_model {cfg.d_model}", f"{cfg.d_model}-d")
    add("attention heads", f"{cfg.n_heads} attention heads", f"{cfg.n_heads} heads")
    add("feed-forward width", f"{cfg.d_ff}")
    add("context length", f"{cfg.max_len}-token", f"{cfg.max_len} tokens")
    add("target dim", f"{cfg.target_dim} dimensions", f"PCA-{cfg.target_dim}", f"{cfg.target_dim}-d")
    add("learning rate", f"{cfg.base_lr:g}", "4e-4", "4 × 10⁻⁴")
    add("warmup", f"{cfg.warmup_steps} warmup")
    add("weight decay", f"weight decay {cfg.weight_decay:g}")
    add("betas", f"({cfg.betas[0]:g}, {cfg.betas[1]:g})")
    add("gradient clip", f"clipping at {cfg.grad_clip:g}", f"clip {cfg.grad_clip:g}")
    add("tokens per step", fmt(cfg.tokens_per_step))
    add("buckets", ", ".join(str(b) for b in cfg.buckets))
    add("EMA momentum", f"{cfg.objective.ema_tau_base:g} to {cfg.objective.ema_tau_final:g}")

    # --- the grid as run --------------------------------------------------- #
    grid = core_grid()
    steps = math.ceil(500_000_000 / TOKENS_PER_STEP)
    add("step budget", fmt(steps) + " steps")
    add("run count", f"{len(grid)} runs")
    seeds = {}
    for r in grid:
        seeds.setdefault(r.run_name, []).append(r.seed)
    tags = {"c1_mlm": "C1", "c2_jepa_ema": "C2", "c3_jepa_frozen": "C3", "c5b_masked_distil": "C5b"}
    for cond, tag in tags.items():
        add(f"{tag} is a condition", tag)
    n3 = sum(1 for s in seeds.values() if len(s) == 3)
    add("seed counts", "seeds 0, 1 and 2" if n3 else "", "three seeds")
    add("C5b seeds", "seeds 0 and 1", "two seeds")

    # --- model -------------------------------------------------------------- #
    enc = Encoder(EncoderConfig(max_len=cfg.max_len))
    n_params = sum(p.numel() for p in enc.parameters())
    add("encoder parameters (exact)", fmt(n_params))
    add("encoder parameters (rounded)", f"{n_params / 1e6:.2f}M")
    pc = PredictorConfig()
    add("predictor layers", f"{pc.n_layers} layers")
    add("predictor width", f"width {pc.d_model}", f"{pc.d_model}")
    add("predictor heads", f"{pc.n_heads} heads")
    add("predictor feed-forward", f"{pc.d_ff}")

    # --- probes -------------------------------------------------------------- #
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    add("linear probe count", f"{words[len(LINEAR_GRID)]} linear", f"{len(LINEAR_GRID)} linear")
    add("MLP probe count", f"{words[len(MLP_GRID)]} one-hidden-layer",
        f"{len(MLP_GRID)} one-hidden-layer", f"{words[len(MLP_GRID)]} MLP")
    add("MLP hidden width", f"{MLP_GRID[0].hidden}-unit", f"{MLP_GRID[0].hidden} units")
    epochs = {c.epochs for c in LINEAR_GRID + MLP_GRID}
    if len(epochs) == 1:
        add("probe epochs", f"{epochs.pop()} epochs")

    # --- data manifests ------------------------------------------------------ #
    ss = json.loads((ROOT / "data/eval/ss/meta.json").read_text())["splits"]
    add("SS train proteins", fmt(ss["train"]["n_sequences"]))
    add("SS valid proteins (TS115)", f"{ss['valid']['n_sequences']}")
    add("SS test proteins (CB513)", f"{ss['test']['n_sequences']}")
    scope = json.loads((ROOT / "data/eval/scope/meta.json").read_text())
    add("SCOPe domains", fmt(scope["splits"]["test"]["n_sequences"]))
    add("SCOPe folds", fmt(scope["n_folds"]))
    add("SCOPe superfamilies", fmt(scope["n_superfamilies"]))

    fr = json.loads((ROOT / "data/splits/filter_report.json").read_text())
    pr = fr["params"]
    add("fetched accessions", fmt(fr["n_input"]))
    add("cluster representatives", fmt(fr["n_cluster_representatives"]))
    add("representatives removed by L0", fmt(fr["n_cluster_representatives"] - fr["n_kept"]))
    add("corpus chains", fmt(fr["n_kept"]))
    add("cluster identity", f"{pr['cluster_identity'] * 100:.0f}% sequence identity",
        f"{pr['cluster_identity'] * 100:.0f}% identity")
    add("cluster coverage", f"{pr['cluster_coverage'] * 100:.0f}%")
    add("leak identity", f"{pr['leak_identity'] * 100:.0f}% identity")
    add("leak coverage", f"{pr['leak_coverage'] * 100:.0f}% coverage")
    return [(c, [s for s in sp if s]) for c, sp in out]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("doc", type=Path, nargs="?", default=ROOT / "paper/methods.md")
    args = p.parse_args(argv)

    text = args.doc.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # drafting notes are not claims
    flat = re.sub(r"\s+", " ", text.replace("**", ""))

    failures = 0
    print(f"checking {args.doc}\n")
    for claim, spellings in facts():
        ok = any(s in flat for s in spellings)
        failures += not ok
        print(f"  {'ok  ' if ok else 'MISS'}  {claim:<34} {' | '.join(spellings)}")

    print("\nstale figures from earlier plans:")
    stale_hits = 0
    for pattern, why in STALE.items():
        for m in re.finditer(pattern, flat):
            ctx = flat[max(0, m.start() - 50): m.end() + 30]
            print(f"  STALE {m.group(0)!r}: {why}\n        ...{ctx}...")
            stale_hits += 1
    if not stale_hits:
        print("  none")

    unfilled = re.findall(r"\[[a-z][^\]]{2,60}\]", flat)
    unfilled = [u for u in unfilled if not re.match(r"\[\d", u)]
    print(f"\nplaceholders left: {len(unfilled)}" + (f"  {unfilled}" if unfilled else ""))

    print(f"\n{failures} missing, {stale_hits} stale")
    return 1 if failures or stale_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
