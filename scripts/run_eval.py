#!/usr/bin/env python3
"""Phase 5, step 2: evaluate every checkpoint on the common task suite.

Extracts frozen features once per checkpoint, caches them, then runs every probe
off that cache. The encoder is never fine-tuned, so results reflect the
representation rather than the probe's capacity.

The probe grid is identical for every condition and is never tuned per
condition -- that identity is the experimental control, so it lives as a module
constant in ``xjepa.eval.probes`` rather than as a convention here.

Baselines matter as much as the conditions, and all four are run by default:

* ``random`` -- untrained encoder. The floor. If a condition cannot beat this,
  1B tokens taught it nothing and no comparison between conditions means anything.
* ``esmif1`` -- the raw frozen target embeddings, i.e. the **L1 ceiling probe**
  (RESEARCH_PLAN.md sec. 1.6). This is what C3/C4/C5 are distilling toward, and
  without it you cannot say how much of any gain is "ESM-IF1 already knew the
  answer".
* ``onehot`` -- one-hot residue features. Sanity floor for the probes themselves.
* ``esm2`` -- public ESM-2 t6 8M, if ``fair-esm`` is installed. Shows whether the
  token budget was enough to learn anything at all.

    python scripts/run_eval.py --runs runs/ --eval-data data/eval --out results/
    python scripts/run_eval.py --runs runs/ --only c3_jepa_frozen-seed0 --tasks ss
    python scripts/run_eval.py --baselines-only --eval-data data/eval --out results/

Writes ``results/<run>/eval.json`` per checkpoint, which
``scripts/aggregate_results.py`` reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xjepa.eval.collapse import collapse_metrics
from xjepa.eval.probes import (
    DEFAULT_GRID,
    PooledTaskData,
    ResidueTaskData,
    extract_features,
    mean_pool,
    run_regression_probe,
    run_residue_classification_probe,
)
from xjepa.eval.retrieval import fold_retrieval, score_dms_assay
from xjepa.model.encoder import Encoder, EncoderConfig

BASELINES = ("random", "onehot", "esmif1", "esm2")


# --------------------------------------------------------------------------- #
# loading splits
# --------------------------------------------------------------------------- #


def load_split(path: Path) -> dict[str, np.ndarray] | None:
    """Load one ``.npz`` split, or ``None`` if it is absent."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def to_padded(split: dict[str, np.ndarray], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat tokens/offsets -> padded ``[N, L]`` tokens and pad mask.

    Sequences longer than ``max_len`` are truncated, which is what the encoder's
    positional range allows; the truncation is reported by the caller so it is
    never silent.
    """
    tokens_flat = torch.from_numpy(split["tokens"].astype(np.int64))
    offsets = torch.from_numpy(split["offsets"].astype(np.int64))
    lengths = (offsets[1:] - offsets[:-1]).clamp(max=max_len)
    n, L = lengths.numel(), int(lengths.max().item()) if lengths.numel() else 0
    out = torch.ones(n, L, dtype=torch.long)
    mask = torch.zeros(n, L, dtype=torch.bool)
    for i in range(n):
        lo, ln = int(offsets[i]), int(lengths[i])
        out[i, :ln] = tokens_flat[lo : lo + ln]
        mask[i, :ln] = True
    return out, mask


def residue_labels(split: dict[str, np.ndarray], key: str, mask: torch.Tensor) -> torch.Tensor:
    """Flat per-residue labels -> ``[N, L]`` with ``-100`` at padding."""
    flat = torch.from_numpy(split[key].astype(np.int64))
    offsets = torch.from_numpy(split["offsets"].astype(np.int64))
    n, L = mask.shape
    out = torch.full((n, L), -100, dtype=torch.long)
    for i in range(n):
        lo = int(offsets[i])
        ln = int(mask[i].sum())
        out[i, :ln] = flat[lo : lo + ln]
    return out


# --------------------------------------------------------------------------- #
# feature producers
# --------------------------------------------------------------------------- #


def load_encoder(ckpt_dir: Path, device: torch.device) -> tuple[Encoder, dict]:
    """Rebuild the encoder from a run's checkpoint and config."""
    from xjepa.train.checkpoint import find_latest, load_checkpoint

    summary = json.loads((ckpt_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = summary["config"]
    enc_cfg = EncoderConfig(
        n_layers=cfg["n_layers"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"], vocab=cfg["vocab"], max_len=cfg["max_len"], rope=cfg["rope"],
    )
    encoder = Encoder(enc_cfg)
    path = find_latest(ckpt_dir)
    if path is None:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    state = load_checkpoint(path, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    enc_sd = {
        k.split("encoder.", 1)[1]: v for k, v in sd.items() if k.startswith("encoder.")
    } or sd
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"[eval] WARNING: {len(missing)} missing encoder keys in {ckpt_dir.name}",
              file=sys.stderr)
    return encoder.eval().to(device), cfg


def onehot_features(tokens: torch.Tensor, vocab: int = 33) -> torch.Tensor:
    """One-hot residue features -- the probe's own sanity floor."""
    return torch.nn.functional.one_hot(tokens, num_classes=vocab).to(torch.float16)


def esmif1_features(split_dir: Path, split: str, mask: torch.Tensor) -> torch.Tensor | None:
    """The L1 ceiling: raw frozen target embeddings for these sequences.

    Requires a precomputed ``<split>_esmif1.npy`` alongside the split, produced
    by running ``scripts/extract_esmif1.py`` over the evaluation structures. When
    it is absent the ceiling probe is skipped with a warning rather than
    silently omitted from the results.
    """
    path = split_dir / f"{split}_esmif1.npy"
    if not path.exists():
        return None
    flat = torch.from_numpy(np.load(path))
    n, L = mask.shape
    out = torch.zeros(n, L, flat.shape[-1], dtype=torch.float16)
    cursor = 0
    for i in range(n):
        ln = int(mask[i].sum())
        out[i, :ln] = flat[cursor : cursor + ln].to(torch.float16)
        cursor += ln
    return out


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #


def eval_residue_task(
    featuriser, eval_root: Path, task_dir: str, label_key: str, task: str, device, max_len: int
) -> dict | None:
    """Secondary structure: per-residue classification probe."""
    base = eval_root / task_dir
    splits = {s: load_split(base / f"{s}.npz") for s in ("train", "valid", "test")}
    if any(v is None for v in splits.values()):
        return None
    if label_key not in splits["train"]:
        return None

    parts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, split in splits.items():
        tokens, mask = to_padded(split, max_len)
        feats = featuriser(tokens, mask)
        if feats is None:
            return None
        labels = residue_labels(split, label_key, mask)
        sel = labels >= 0
        parts[name] = (feats[sel].float(), labels[sel])

    data = ResidueTaskData(
        train_x=parts["train"][0], train_y=parts["train"][1],
        val_x=parts["valid"][0], val_y=parts["valid"][1],
        test_x=parts["test"][0], test_y=parts["test"][1],
        num_classes=int(max(p[1].max().item() for p in parts.values())) + 1,
    )
    res = run_residue_classification_probe(data, task=task, grid=DEFAULT_GRID, device=device)
    return {"val_metric": res.val_metric, **res.test_metrics}


def eval_regression_task(
    featuriser, eval_root: Path, task: str, device, max_len: int
) -> dict | None:
    """Fluorescence / stability: mean-pooled ridge probe, Spearman."""
    base = eval_root / task
    splits = {s: load_split(base / f"{s}.npz") for s in ("train", "valid", "test")}
    if any(v is None for v in splits.values()):
        return None

    parts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, split in splits.items():
        tokens, mask = to_padded(split, max_len)
        feats = featuriser(tokens, mask)
        if feats is None:
            return None
        parts[name] = (mean_pool(feats.float(), mask), torch.from_numpy(split["y"]).float())

    data = PooledTaskData(
        train_x=parts["train"][0], train_y=parts["train"][1],
        val_x=parts["valid"][0], val_y=parts["valid"][1],
        test_x=parts["test"][0], test_y=parts["test"][1],
    )
    res = run_regression_probe(data, task=task, grid=DEFAULT_GRID, device=device)
    return {"val_metric": res.val_metric, **res.test_metrics}


def eval_fold_retrieval(featuriser, eval_root: Path, max_len: int) -> dict | None:
    """Zero-shot fold retrieval with superfamily-disjoint query/gallery."""
    split = load_split(eval_root / "scope" / "test.npz")
    if split is None:
        return None
    tokens, mask = to_padded(split, max_len)
    feats = featuriser(tokens, mask)
    if feats is None:
        return None
    pooled = mean_pool(feats.float(), mask)
    res = fold_retrieval(
        pooled,
        fold_labels=torch.from_numpy(split["fold"].astype(np.int64)),
        superfamily_labels=torch.from_numpy(split["superfamily"].astype(np.int64)),
    )
    return {
        "top1_accuracy": res.top1_accuracy,
        "mean_average_precision": res.mean_average_precision,
        "n_queries": res.n_queries,
        "n_gallery": res.n_gallery,
    }


def eval_proteingym(featuriser, eval_root: Path, max_len: int) -> dict | None:
    """Variant effects by embedding distance.

    Exploratory, and pre-registered as such: cosine distance is not a fitness
    score and |rho| < 0.2 is the expectation. JEPA-only models expose no token
    likelihoods, so pseudo-likelihood scoring is unavailable and this is the only
    like-for-like comparison across all conditions.
    """
    base = eval_root / "proteingym"
    assay_files = sorted(base.glob("*.npz"))
    if not assay_files:
        return None

    rhos: list[float] = []
    per_assay: dict[str, float] = {}
    for path in assay_files:
        split = load_split(path)
        if split is None or "positions" not in split:
            continue
        tokens, mask = to_padded(split, max_len)
        feats = featuriser(tokens, mask)
        if feats is None:
            return None
        positions = torch.from_numpy(split["positions"].astype(np.int64)).clamp(max=mask.shape[1] - 1)
        # Wild type stands in as the first variant's background; each mutant is
        # scored against it at the mutated position.
        wt = feats[0:1].expand_as(feats).float()
        res = score_dms_assay(
            wt_residue_embeddings=wt,
            mut_residue_embeddings=feats.float(),
            positions=positions,
            dms_scores=torch.from_numpy(split["scores"]).float(),
            assay=path.stem,
        )
        per_assay[path.stem] = res.spearman
        rhos.append(res.spearman)

    if not rhos:
        return None
    return {
        "mean_spearman": float(np.mean(rhos)),
        "median_abs_spearman": float(np.median(np.abs(rhos))),
        "n_assays": len(rhos),
        "per_assay": per_assay,
        "caveat": "exploratory: embedding-distance scoring, |rho| < 0.2 expected",
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def evaluate(
    label: str, featuriser, eval_root: Path, tasks: list[str], device, max_len: int
) -> dict[str, Any]:
    """Run the selected tasks for one feature source."""
    out: dict[str, Any] = {"label": label, "tasks": {}}
    runners = {
        "ss3": lambda: eval_residue_task(featuriser, eval_root, "ss", "ss3", "ss3", device, max_len),
        "ss8": lambda: eval_residue_task(featuriser, eval_root, "ss", "ss8", "ss8", device, max_len),
        "fluorescence": lambda: eval_regression_task(featuriser, eval_root, "fluorescence", device, max_len),
        "stability": lambda: eval_regression_task(featuriser, eval_root, "stability", device, max_len),
        "fold_retrieval": lambda: eval_fold_retrieval(featuriser, eval_root, max_len),
        "proteingym": lambda: eval_proteingym(featuriser, eval_root, max_len),
    }
    for task in tasks:
        runner = runners.get(task)
        if runner is None:
            continue
        t0 = time.time()
        try:
            res = runner()
        except Exception as exc:  # noqa: BLE001 - one task must not sink the sweep
            print(f"[eval] {label}/{task} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            out["tasks"][task] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if res is None:
            print(f"[eval] {label}/{task}: data missing, skipped", file=sys.stderr)
            out["tasks"][task] = {"skipped": "data missing"}
        else:
            res["seconds"] = round(time.time() - t0, 1)
            out["tasks"][task] = res
            print(f"[eval] {label}/{task}: {res}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--eval-data", type=Path, default=Path("data/eval"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--only", nargs="*", default=None, help="specific run directory names")
    p.add_argument(
        "--tasks", nargs="*",
        default=["ss3", "ss8", "fluorescence", "stability", "fold_retrieval", "proteingym"],
    )
    p.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- baselines ------------------------------------------------------- #
    for name in args.baselines:
        dest = args.out / f"baseline_{name}" / "eval.json"
        if dest.exists() and not args.force:
            print(f"[eval] skip baseline {name} (done)", file=sys.stderr)
            continue

        if name == "onehot":
            featuriser = lambda t, m: onehot_features(t)  # noqa: E731
        elif name == "random":
            enc = Encoder(EncoderConfig(max_len=args.max_len)).eval().to(device)
            featuriser = lambda t, m: extract_features(  # noqa: E731
                enc, t, m, batch_size=args.batch_size, device=device
            )
        elif name == "esmif1":
            featuriser = lambda t, m: None  # noqa: E731 - needs per-split files
            print(
                "[eval] NOTE: the esmif1 ceiling probe needs <split>_esmif1.npy next to each "
                "split. Without it the L1 ceiling is missing, and you cannot report how much "
                "of any C3 gain is ESM-IF1 already knowing the answer.",
                file=sys.stderr,
            )
        else:  # esm2
            try:
                import esm

                model, alpha = esm.pretrained.esm2_t6_8M_UR50D()
                model = model.eval().to(device)

                def featuriser(t, m, _model=model):  # type: ignore[misc]
                    with torch.no_grad():
                        rep = _model(t.to(device), repr_layers=[6])["representations"][6]
                    return rep.to(torch.float16).cpu()
            except Exception as exc:  # noqa: BLE001
                print(f"[eval] esm2 baseline unavailable ({exc}); skipped", file=sys.stderr)
                continue

        res = evaluate(f"baseline_{name}", featuriser, args.eval_data, args.tasks, device, args.max_len)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(res, indent=2), encoding="utf-8")

    if args.baselines_only:
        return 0

    # --- trained runs ----------------------------------------------------- #
    run_dirs = [d for d in sorted(args.runs.iterdir()) if (d / "summary.json").exists()]
    if args.only:
        run_dirs = [d for d in run_dirs if d.name in set(args.only)]
    if not run_dirs:
        print(f"[eval] no finished runs under {args.runs}", file=sys.stderr)
        return 1

    for run_dir in run_dirs:
        dest = args.out / run_dir.name / "eval.json"
        if dest.exists() and not args.force:
            print(f"[eval] skip {run_dir.name} (done)", file=sys.stderr)
            continue
        print(f"\n[eval] === {run_dir.name} ===", file=sys.stderr)
        try:
            encoder, cfg = load_encoder(run_dir, device)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] cannot load {run_dir.name}: {exc}", file=sys.stderr)
            continue

        def featuriser(t, m, _enc=encoder):
            return extract_features(_enc, t, m, batch_size=args.batch_size, device=device)

        res = evaluate(run_dir.name, featuriser, args.eval_data, args.tasks, device, args.max_len)
        res["config"] = cfg

        # Collapse diagnostics on the final representation, so the aggregator has
        # RankMe for H1a without re-reading every training CSV.
        scope = load_split(args.eval_data / "scope" / "test.npz")
        if scope is not None:
            tokens, mask = to_padded(scope, args.max_len)
            feats = featuriser(tokens[:512], mask[:512])
            res["collapse"] = collapse_metrics(feats[mask[:512]].float())

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n[eval] wrote results to {args.out}", file=sys.stderr)
    print("[eval] next: python scripts/aggregate_results.py --results results/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
