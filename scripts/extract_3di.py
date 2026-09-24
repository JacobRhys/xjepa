#!/usr/bin/env python3
"""Phase 3 alternative: Foldseek 3Di structural tokens as a second target.

Foldseek's 3Di alphabet is a 20-letter structural vocabulary derived from
backbone geometry. It costs CPU-minutes and about 10 MB for the whole corpus,
against ESM-IF1's 3-4 GPU-hours and 12.8 GB.

Two reasons to build it, and the second one is the real one:

1. **Fallback.** If the ``fair-esm`` / ``torch-geometric`` stack defeats you, or
   the ESM-IF1 extraction overruns the budget, C3 becomes cross-entropy over
   3Di tokens at masked positions and the study still runs.
2. **Robustness check.** Running C3 against both targets tells you whether any
   result is a property of latent structure prediction or an artefact of one
   particular structure encoder. That is worth having even when ESM-IF1 works,
   and at this price there is no reason not to.

Note the difference in kind: 3Di gives a *discrete* target (cross-entropy over
20 classes), ESM-IF1 a *continuous* one (regression in 512-d). A 3Di-target C3
is closer in form to MLM than the ESM-IF1 version is, which weakens the "this
is not disguised token prediction" argument -- so treat it as a check on the
main result, not a replacement for it.

Requires Foldseek on PATH. It is **not** in homebrew-core; upstream ships
official binaries::

    # macOS (universal), Linux builds also at https://mmseqs.com/foldseek/
    curl -fsSL https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz | tar xz
    ln -s "$PWD/foldseek/bin/foldseek" /opt/homebrew/bin/foldseek

    # or via conda
    conda install -c conda-forge -c bioconda foldseek

Then::

    python scripts/extract_3di.py --shards data/structures \\
        --allowlist data/splits/pretrain_accessions.txt --out data/raw_3di
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xjepa.data.alphabet import encode

#: Foldseek's 3Di structural alphabet.
THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREEDI_TO_ID = {c: i for i, c in enumerate(THREEDI_ALPHABET)}
#: Reserved id for a residue Foldseek could not assign.
THREEDI_UNK = len(THREEDI_ALPHABET)

_PDB_ATOM = (
    "ATOM  {serial:5d} {name:^4s}{alt:1s}{resname:>3s} {chain:1s}{resseq:4d}{icode:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}\n"
)


def require_foldseek() -> str:
    exe = shutil.which("foldseek")
    if exe is None:
        print(
            "foldseek not found on PATH. It is NOT in homebrew-core; use the\n"
            "official upstream binary or conda:\n"
            "  curl -fsSL https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz | tar xz\n"
            "  ln -s \"$PWD/foldseek/bin/foldseek\" /opt/homebrew/bin/foldseek\n"
            "  (Linux builds: https://mmseqs.com/foldseek/)\n"
            "  conda:  conda install -c conda-forge -c bioconda foldseek",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return exe


def write_pdb(path: Path, seq: str, coords: np.ndarray) -> None:
    """Write a minimal N/CA/C backbone PDB.

    Foldseek derives 3Di states from backbone geometry, so N, CA and C suffice;
    it does not need side chains or a CB.
    """
    from xjepa.data.alphabet import THREE_TO_ONE

    one_to_three = {v: k for k, v in THREE_TO_ONE.items() if len(k) == 3}
    serial = 1
    with open(path, "w", encoding="utf-8") as fh:
        for i, aa in enumerate(seq):
            resname = one_to_three.get(aa, "GLY")
            for j, atom in enumerate(("N", "CA", "C")):
                x, y, z = (float(v) for v in coords[i, j])
                fh.write(
                    _PDB_ATOM.format(
                        serial=serial, name=f" {atom:<3s}"[:4], alt=" ", resname=resname,
                        chain="A", resseq=i + 1, icode=" ", x=x, y=y, z=z, occ=1.0, bfac=0.0,
                    )
                )
                serial += 1
        fh.write("TER\nEND\n")


def run_foldseek(exe: str, pdb_dir: Path, out_tsv: Path) -> None:
    """Convert a directory of PDBs to 3Di descriptors.

    ``structureto3didescriptor`` writes one line per structure:
    ``name<TAB>amino-acid sequence<TAB>3Di sequence<TAB>...``
    """
    proc = subprocess.run(
        [exe, "structureto3didescriptor", str(pdb_dir), str(out_tsv), "--threads", "8"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"[3di] foldseek failed:\n{proc.stderr[-4000:]}", file=sys.stderr)
        raise SystemExit(proc.returncode)


def parse_3di(tsv: Path) -> dict[str, str]:
    """Map structure name -> 3Di sequence."""
    out: dict[str, str] = {}
    with open(tsv, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                name = Path(parts[0]).stem
                out[name] = parts[2].strip().upper()
    return out


def encode_3di(s: str) -> np.ndarray:
    """Encode a 3Di string to ids, with unassigned states mapped to THREEDI_UNK."""
    return np.fromiter((THREEDI_TO_ID.get(c, THREEDI_UNK) for c in s), dtype=np.uint8, count=len(s))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allowlist", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=2000, help="PDBs per foldseek invocation")
    args = p.parse_args(argv)

    exe = require_foldseek()
    args.out.mkdir(parents=True, exist_ok=True)

    allow: set[str] | None = None
    if args.allowlist:
        allow = {
            l.strip() for l in args.allowlist.read_text(encoding="utf-8").splitlines() if l.strip()
        }

    # Reuse the shard iterator from the ESM-IF1 path so both targets are built
    # over exactly the same chains in exactly the same order. Works whether this
    # is run as `python scripts/extract_3di.py` or imported as `scripts.extract_3di`.
    try:
        from extract_esmif1 import iter_chains
    except ImportError:
        from scripts.extract_esmif1 import iter_chains

    tokens_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    accessions: list[str] = []
    lengths: list[int] = []
    n_mismatch = 0

    pending: list[tuple[str, str, np.ndarray]] = []

    def flush(batch: list[tuple[str, str, np.ndarray]]) -> None:
        nonlocal n_mismatch
        if not batch:
            return
        tmp = Path(tempfile.mkdtemp(prefix="xjepa_3di_"))
        try:
            pdb_dir = tmp / "pdb"
            pdb_dir.mkdir()
            for acc, seq, coords in batch:
                write_pdb(pdb_dir / f"{acc}.pdb", seq, coords)
            tsv = tmp / "out.tsv"
            run_foldseek(exe, pdb_dir, tsv)
            mapping = parse_3di(tsv)
            for acc, seq, _ in batch:
                s3 = mapping.get(acc)
                if s3 is None:
                    n_mismatch += 1
                    continue
                if len(s3) != len(seq):
                    # Alignment between residue and structural token must be
                    # exact, or every target is paired with the wrong residue.
                    n_mismatch += 1
                    continue
                accessions.append(acc)
                lengths.append(len(seq))
                tokens_all.append(encode(seq))
                targets_all.append(encode_3di(s3))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    for item in iter_chains(args.shards, allow, args.limit):
        pending.append(item)
        if len(pending) >= args.batch:
            flush(pending)
            print(f"[3di] {len(accessions):,} chains done", file=sys.stderr)
            pending = []
    flush(pending)

    if not accessions:
        print("[3di] nothing produced", file=sys.stderr)
        return 1

    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    np.save(args.out / "tokens.npy", np.concatenate(tokens_all))
    np.save(args.out / "offsets.npy", offsets.astype(np.int32))
    np.save(args.out / "targets_3di.npy", np.concatenate(targets_all))
    (args.out / "accessions.txt").write_text("\n".join(accessions) + "\n", encoding="utf-8")

    meta = {
        "n_chains": len(accessions),
        "n_residues": int(offsets[-1]),
        "n_dropped_mismatch": n_mismatch,
        "alphabet": THREEDI_ALPHABET,
        "unk_id": THREEDI_UNK,
        "target_kind": "discrete (cross-entropy over 20 structural states)",
        "note": (
            "A 3Di-target C3 is closer in form to MLM than the ESM-IF1 version. "
            "Use it as a robustness check on the continuous-target result, not a "
            "replacement for it."
        ),
    }
    (args.out / "extract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\n[3di] {len(accessions):,} chains, {int(offsets[-1]):,} residues -> {args.out}\n"
        f"[3di] dropped {n_mismatch:,} on length mismatch or missing output",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
