"""Phase 2/3 pipeline: alphabet, PDB parsing, shard IO.

The parser tests matter more than they look. Residue-to-target alignment is the
one thing in this pipeline that can be wrong *silently*: a one-residue shift
pairs every sequence position with its neighbour's structure target, training
proceeds normally, losses fall, and every downstream number is quietly garbage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.extract_3di import encode_3di, write_pdb, THREEDI_UNK
from scripts.fetch_afdb import Chain, keep, parse_backbone, write_shard
from xjepa.data.alphabet import (
    MASK_ID,
    PAD_ID,
    TOKENS,
    UNK_ID,
    VOCAB_SIZE,
    encode,
    encode_many,
)


# --------------------------------------------------------------------------- #
# alphabet
# --------------------------------------------------------------------------- #


def test_alphabet_matches_esm2() -> None:
    """Token ids must match ESM-2's, or the baseline comparison is meaningless."""
    assert VOCAB_SIZE == 33
    assert PAD_ID == 1 and MASK_ID == 32
    assert TOKENS[0] == "<cls>" and TOKENS[32] == "<mask>"
    # the 20 standard residues all live in 4..23
    for aa in "LAGVSERTIDPKQNFYMHWC":
        assert 4 <= TOKENS.index(aa) <= 23


def test_encode_unknown_is_unk_not_a_crash() -> None:
    """AFDB models carry odd residues; they must not abort a 50k-chain run."""
    out = encode("AC?G")
    assert out[2] == UNK_ID
    assert out.dtype == np.uint8


def test_encode_many_builds_flat_corpus_layout() -> None:
    seqs = ["ACD", "", "WYV", "M"]
    tokens, offsets = encode_many(seqs)
    assert offsets.dtype == np.int32
    assert offsets.tolist() == [0, 3, 3, 6, 7]
    assert tokens.shape == (7,)
    for i, s in enumerate(seqs):
        lo, hi = offsets[i], offsets[i + 1]
        assert np.array_equal(tokens[lo:hi], encode(s))


# --------------------------------------------------------------------------- #
# PDB parsing
# --------------------------------------------------------------------------- #


def _synthetic_pdb(seq: str, plddt: float = 90.0, drop_atom: tuple[int, str] | None = None) -> str:
    """Build a backbone PDB with known coordinates: residue i at x = i."""
    from xjepa.data.alphabet import THREE_TO_ONE

    one_to_three = {v: k for k, v in THREE_TO_ONE.items() if len(k) == 3}
    lines = []
    serial = 1
    for i, aa in enumerate(seq):
        resname = one_to_three.get(aa, "GLY")
        for j, atom in enumerate(("N", "CA", "C")):
            if drop_atom == (i, atom):
                continue
            x, y, z = float(i), float(j), 0.0
            lines.append(
                f"ATOM  {serial:5d} {atom:<4s} {resname:>3s} A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines)


def test_parse_backbone_roundtrip() -> None:
    seq = "ACDEFGHIKL"
    chain = parse_backbone(_synthetic_pdb(seq), "P00001")
    assert chain is not None
    assert chain.seq == seq
    assert chain.coords.shape == (len(seq), 3, 3)
    assert chain.plddt.shape == (len(seq),)
    # residue i was written at x = i, for every backbone atom
    assert np.allclose(chain.coords[:, :, 0], np.arange(len(seq))[:, None].astype(np.float32))
    assert chain.mean_plddt == pytest.approx(90.0)


def test_incomplete_residue_drops_sequence_position_too() -> None:
    """Dropping a residue's coordinates must drop its letter, or everything shifts."""
    seq = "ACDEFG"
    chain = parse_backbone(_synthetic_pdb(seq, drop_atom=(2, "CA")), "P2")
    assert chain is not None
    assert chain.seq == "ACEFG"  # the D is gone
    assert chain.coords.shape == (5, 3, 3)
    # and the surviving coordinates still carry their ORIGINAL x indices
    assert chain.coords[:, 0, 0].tolist() == [0.0, 1.0, 3.0, 4.0, 5.0]


def test_parse_backbone_empty_input() -> None:
    assert parse_backbone("HEADER nothing here\nEND\n", "P3") is None


def test_msds_selenomethionine_reads_as_met() -> None:
    pdb = (
        "ATOM      1  N   MSE A   1       0.000   0.000   0.000  1.00 88.00\n"
        "ATOM      2  CA  MSE A   1       0.000   1.000   0.000  1.00 88.00\n"
        "ATOM      3  C   MSE A   1       0.000   2.000   0.000  1.00 88.00\n"
    )
    chain = parse_backbone(pdb, "P4")
    assert chain is not None and chain.seq == "M"


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #


def _chain(seq: str, plddt: float) -> Chain:
    n = len(seq)
    return Chain(
        accession="X",
        seq=seq,
        coords=np.zeros((n, 3, 3), dtype=np.float32),
        plddt=np.full(n, plddt, dtype=np.float32),
    )


@pytest.mark.parametrize(
    "length,plddt,expected",
    [
        (100, 90.0, True),
        (10, 90.0, False),   # too short
        (600, 90.0, False),  # too long
        (100, 50.0, False),  # disordered: noisy, near-degenerate targets
    ],
)
def test_filters(length: int, plddt: float, expected: bool) -> None:
    assert keep(_chain("A" * length, plddt), 40, 512, 70.0) is expected


# --------------------------------------------------------------------------- #
# shard IO
# --------------------------------------------------------------------------- #


def test_shard_roundtrip(tmp_path: Path) -> None:
    chains = [
        parse_backbone(_synthetic_pdb("ACDEF"), "P1"),
        parse_backbone(_synthetic_pdb("WYVMH"), "P2"),
        parse_backbone(_synthetic_pdb("GGG"), "P3"),
    ]
    path = tmp_path / "shard_0000.npz"
    write_shard(path, [c for c in chains if c is not None])
    assert not list(tmp_path.glob("*.tmp.npz")), "temp file left behind"

    with np.load(path, allow_pickle=True) as z:
        accs = [str(a) for a in z["accessions"]]
        seqs = [str(s) for s in z["seqs"]]
        offsets, coords = z["offsets"], z["coords"]

    assert accs == ["P1", "P2", "P3"]
    assert offsets.tolist() == [0, 5, 10, 13]
    assert coords.shape == (13, 3, 3)
    for i, seq in enumerate(seqs):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        assert hi - lo == len(seq), "offsets disagree with sequence length"


def test_shard_iteration_matches_written_order(tmp_path: Path) -> None:
    """extract_esmif1 and extract_3di must see chains in identical order."""
    from scripts.extract_esmif1 import count_chains, iter_chains

    chains = [parse_backbone(_synthetic_pdb(s), f"P{i}") for i, s in enumerate(["AC", "DEF", "GH"])]
    write_shard(tmp_path / "shard_0000.npz", [c for c in chains if c is not None])

    got = [(acc, seq, c.shape[0]) for acc, seq, c in iter_chains(tmp_path, None, None)]
    assert [g[0] for g in got] == ["P0", "P1", "P2"]
    assert [g[1] for g in got] == ["AC", "DEF", "GH"]
    assert [g[2] for g in got] == [2, 3, 2]

    assert count_chains(tmp_path, None, None) == (3, 7)
    assert count_chains(tmp_path, {"P1"}, None) == (1, 3)
    assert count_chains(tmp_path, None, 2) == (2, 5)


# --------------------------------------------------------------------------- #
# 3Di
# --------------------------------------------------------------------------- #


def test_encode_3di_unassigned_state() -> None:
    out = encode_3di("ACD#")
    assert out.tolist() == [0, 1, 2, THREEDI_UNK]


def test_write_pdb_is_reparseable(tmp_path: Path) -> None:
    """The PDB we hand Foldseek must parse back to the same residues."""
    seq = "ACDEFGHIKL"
    coords = np.arange(len(seq) * 9, dtype=np.float32).reshape(len(seq), 3, 3)
    path = tmp_path / "x.pdb"
    write_pdb(path, seq, coords)

    chain = parse_backbone(path.read_text(encoding="utf-8"), "x")
    assert chain is not None
    assert chain.seq == seq
    assert np.allclose(chain.coords, coords, atol=1e-3)
