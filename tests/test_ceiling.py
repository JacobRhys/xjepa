"""L1 ceiling: structure parsing, gap placement, and per-split feature lookup.

The ceiling is only meaningful if each feature row lines up with the residue the
probe labels. An off-by-one here pairs every residue with its neighbour's
structure and inflates or deflates the ceiling silently, so the alignment is
tested on inputs where the right answer is known.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_ceiling_features import (
    esmfold_safe, parse_pdbstyle, place_coords, split_sequences)
from scripts.run_eval import esmif1_features, featurise, to_padded
from xjepa.data.alphabet import encode_many


def _atom(serial: int, name: str, res: str, resseq: int, x: float, het: bool = False) -> str:
    rec = "HETATM" if het else "ATOM  "
    return (f"{rec}{serial:5d} {name:<4} {res} A{resseq:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00")


def test_parse_pdbstyle_keeps_mse_and_drops_incomplete_residues() -> None:
    lines = []
    for i, (res, het) in enumerate([("GLY", False), ("MSE", True), ("ALA", False)]):
        for j, a in enumerate(("N", "CA", "C")):
            if res == "ALA" and a == "C":
                continue  # incomplete backbone: must be dropped
            lines.append(_atom(i * 3 + j, a, res, i + 1, float(i * 10 + j), het))
    seq, coords = parse_pdbstyle("\n".join(lines))
    assert seq == "GM"
    assert coords.shape == (2, 3, 3)
    assert coords[1, 1, 0] == pytest.approx(11.0)  # MSE's CA


def test_place_coords_leaves_unresolved_residues_nan() -> None:
    coords = np.arange(3 * 9, dtype=np.float32).reshape(3, 3, 3)
    out = place_coords("MKVLA", "KVA", coords)
    finite = np.isfinite(out).all(axis=(1, 2))
    assert finite.tolist() == [False, True, True, False, True]
    np.testing.assert_array_equal(out[4], coords[2])


def test_split_sequences_round_trips() -> None:
    seqs = ["MKV", "ACDEFGHIK"]
    tokens, offsets = encode_many(seqs)
    assert split_sequences({"tokens": tokens, "offsets": offsets}) == seqs


def test_ceiling_features_align_with_truncated_windows(tmp_path: Path) -> None:
    seqs = ["MKVLA", "ACD"]
    tokens, offsets = encode_many(seqs)
    max_len = 4
    split = {"tokens": tokens, "offsets": offsets}
    toks, mask = to_padded(split, max_len)
    # Row value = (protein, position), so misalignment is visible.
    rows = [[p * 10 + i] * 2 for p, s in enumerate(seqs) for i in range(min(len(s), max_len))]
    np.save(tmp_path / "test_esmif1.npy", np.asarray(rows, dtype=np.float16))
    (tmp_path / "test_esmif1.json").write_text(json.dumps({"max_len": max_len}))

    feats = featurise(esmif1_features, toks, mask, tmp_path, "test")
    assert feats.shape == (2, 4, 2)
    assert feats[0, 3, 0] == 3 and feats[1, 2, 0] == 12
    assert feats[1, 3, 0] == 0  # padding


def test_ceiling_features_refuse_a_mismatched_window(tmp_path: Path) -> None:
    tokens, offsets = encode_many(["MKVLA"])
    toks, mask = to_padded({"tokens": tokens, "offsets": offsets}, 5)
    np.save(tmp_path / "test_esmif1.npy", np.zeros((4, 2), np.float16))
    (tmp_path / "test_esmif1.json").write_text(json.dumps({"max_len": 4}))
    with pytest.raises(ValueError):
        featurise(esmif1_features, toks, mask, tmp_path, "test")


def test_missing_ceiling_file_reports_data_missing(tmp_path: Path) -> None:
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert featurise(esmif1_features, torch.zeros(1, 3, dtype=torch.long), mask,
                     tmp_path, "test") is None


def test_subsampled_ceiling_marks_uncovered_proteins_nan(tmp_path: Path) -> None:
    seqs = ["MKV", "ACDE", "GH"]
    tokens, offsets = encode_many(seqs)
    toks, mask = to_padded({"tokens": tokens, "offsets": offsets}, 8)
    # Only proteins 0 and 2 were folded.
    np.save(tmp_path / "train_esmif1_index.npy", np.array([0, 2]))
    np.save(tmp_path / "train_esmif1.npy", np.arange(5 * 2, dtype=np.float16).reshape(5, 2))
    feats = featurise(esmif1_features, toks, mask, tmp_path, "train")
    assert torch.isfinite(feats[0, :3]).all() and torch.isfinite(feats[2, :2]).all()
    assert torch.isnan(feats[1, :4]).all()
    assert feats[2, 0, 0] == 6  # protein 2 starts after protein 0's three rows
    assert torch.isfinite(feats[~mask]).all()  # padding stays finite


def test_esmfold_safe_maps_rare_residues_to_x_in_place() -> None:
    # SS test protein 149 carries a U; ESMFold's tokenizer has no U and crashed.
    assert esmfold_safe("MKUVZBOX") == "MKXVXXXX"
