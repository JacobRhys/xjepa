"""The trained-run path of run_eval: checkpoint -> encoder -> probes -> eval.json.

The baselines never touch a checkpoint, so they can all pass while every trained
run fails to load. That happened on the first v3 pod: all 11 runs were skipped
with "cannot load" and the session ended with only baselines evaluated. These
tests write a checkpoint the way the trainer does and drive `run_eval.main` over
it on tiny synthetic data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import run_eval
from xjepa.train.checkpoint import TrainState, save_checkpoint
from xjepa.train.trainer import RunConfig, build_model, build_objective
from xjepa.train.objectives import ObjectiveConfig

TINY = dict(n_layers=2, d_model=32, n_heads=4, d_ff=64, target_dim=16, max_len=64)


def _write_run(root: Path, name: str = "c3_jepa_frozen") -> tuple[Path, torch.nn.Module]:
    cfg = replace(
        RunConfig(name=name, objective=ObjectiveConfig(name=name), device="cpu",
                  compile=False, bf16=False),
        **TINY,
    )
    model = build_model(cfg, build_objective(cfg.objective))
    opt = torch.optim.AdamW(model.parameters())
    run_dir = root / f"{name}-seed0"
    save_checkpoint(run_dir, model, opt, TrainState(step=10, tokens_seen=100, seed=0),
                    cfg.to_dict())
    (run_dir / "summary.json").write_text(
        json.dumps({"name": name, "config": cfg.to_dict()}), encoding="utf-8")
    return run_dir, model


def _split(rng: np.random.Generator, n: int, **labels) -> dict[str, np.ndarray]:
    lens = rng.integers(20, 40, size=n)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    out = {"tokens": rng.integers(4, 24, size=offsets[-1]).astype(np.uint8), "offsets": offsets}
    for key, n_classes in labels.items():
        out[key] = rng.integers(0, n_classes, size=offsets[-1]).astype(np.int64)
    return out


def _write_eval_data(root: Path) -> None:
    rng = np.random.default_rng(0)
    (root / "ss").mkdir(parents=True)
    for split, n in (("train", 12), ("valid", 4), ("test", 4)):
        np.savez(root / "ss" / f"{split}.npz", **_split(rng, n, ss3=3, ss8=8))
    (root / "scope").mkdir()
    scope = _split(rng, 24)
    scope["fold"] = np.repeat(np.arange(6), 4)
    scope["superfamily"] = np.arange(24) // 2
    np.savez(root / "scope" / "test.npz", **scope)


def test_load_encoder_restores_the_trained_weights(tmp_path: Path) -> None:
    run_dir, model = _write_run(tmp_path)
    encoder, cfg = run_eval.load_encoder(run_dir, torch.device("cpu"))
    assert cfg["d_model"] == TINY["d_model"]
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, model.encoder.state_dict()[key])


def test_load_encoder_refuses_a_partial_load(tmp_path: Path) -> None:
    """A checkpoint without encoder weights must fail, not evaluate a random encoder."""
    run_dir, _ = _write_run(tmp_path)
    ckpt = next(run_dir.glob("ckpt_*.pt"))
    payload = torch.load(ckpt, weights_only=False)
    payload["model"] = {k: v for k, v in payload["model"].items() if not k.startswith("encoder.")}
    torch.save(payload, ckpt)
    with pytest.raises(RuntimeError):
        run_eval.load_encoder(run_dir, torch.device("cpu"))


def test_main_evaluates_trained_runs_end_to_end(tmp_path: Path) -> None:
    runs, data, results = tmp_path / "runs", tmp_path / "eval", tmp_path / "results"
    _write_run(runs)
    _write_eval_data(data)
    rc = run_eval.main([
        "--runs", str(runs), "--eval-data", str(data), "--out", str(results),
        "--tasks", "ss3", "ss8", "fold_retrieval", "--baselines", "--device", "cpu",
        "--max-len", str(TINY["max_len"]), "--batch-size", "8",
    ])
    assert rc == 0
    res = json.loads((results / "c3_jepa_frozen-seed0" / "eval.json").read_text())
    for task in ("ss3", "ss8", "fold_retrieval"):
        assert "error" not in res["tasks"][task], res["tasks"][task]
        assert "skipped" not in res["tasks"][task]
    assert "rankme" in res["collapse"]
