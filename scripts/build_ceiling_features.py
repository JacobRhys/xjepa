#!/usr/bin/env python3
"""Phase 5, step 1: ESM-IF1 features for the evaluation proteins (the L1 ceiling).

The L1 ceiling probe (RESEARCH_PLAN.md sec. 1.6) trains the same frozen-feature
probes on ESM-IF1's own embeddings. It is what C3/C4/C5 distil toward, so it
bounds how much of any C3 gain is "ESM-IF1 already knew the answer". ESM-IF1
reads backbone coordinates, which most evaluation sets do not ship, so each
structure task gets the best structure source available:

* ``scope`` -- the real ASTRAL 2.08 40% PDB-style files, the same domains the
  retrieval benchmark uses. Read straight from the tarball, never extracted.
* ``ss``, ``contact`` -- predicted by ESMFold. CB513/TS115/NetSurfP carry no
  usable structure, and ProteinNet stores only C-alpha while ESM-IF1 needs
  N/CA/C. ESM-IF1 was itself trained largely on predicted structures, so this
  is in-distribution for it. Disclose it: the ceiling then measures what
  ESMFold + ESM-IF1 know together, and ESMFold is a 3B-parameter sequence model.

Fitness tasks (fluorescence, stability, ProteinGym) get no ceiling -- the plan
expects structure targets to carry little fitness signal there, and folding
~100k near-identical GFP and mini-protein variants buys nothing. They are
reported as out of scope rather than silently missing.

Features are projected through the corpus PCA basis (``data/corpus/pca_*.npy``),
so the probe sees exactly the 128-d standardised space C3 regresses onto.
Each protein contributes its first ``--max-len`` residues -- the window
``run_eval.py`` gives the encoders -- and its structure is that window's.

Output, next to each split: ``<split>_esmif1.npy`` (flat ``[sum(min(L, max_len)), 128]``
fp16, in split order) and ``<split>_esmif1.json``. Work is saved in chunks, so a
killed run resumes where it stopped.

    python scripts/build_ceiling_features.py --eval-data data/eval --corpus data/corpus \\
        --scope-tarball pdbstyle-sel-gs-bib-40-2.08.tgz

Install (on top of the ESM-IF1 stack): ``pip install transformers accelerate``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xjepa.data.alphabet import THREE_TO_ONE, TOKENS

SCOPE_TARBALL_URL = (
    "https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.08.tgz"
)
STRUCTURE_SOURCE = {"ss": "esmfold", "contact": "esmfold", "scope": "astral-2.08-pdbstyle"}
SPLITS = {"ss": ("train", "valid", "test"), "contact": ("train", "valid", "test"),
          "scope": ("test",)}
BACKBONE = ("N", "CA", "C")


# --------------------------------------------------------------------------- #
# sequences
# --------------------------------------------------------------------------- #


def split_sequences(split: dict[str, np.ndarray]) -> list[str]:
    """Decode a flat tokens/offsets split back to sequence strings."""
    offsets = split["offsets"].astype(np.int64)
    toks = split["tokens"]
    return [
        "".join(TOKENS[int(t)] if len(TOKENS[int(t)]) == 1 else "X" for t in toks[lo:hi])
        for lo, hi in zip(offsets[:-1], offsets[1:])
    ]


def fasta_ids(path: Path) -> list[str]:
    return [ln[1:].split()[0] for ln in path.read_text().splitlines() if ln.startswith(">")]


# --------------------------------------------------------------------------- #
# structure sources
# --------------------------------------------------------------------------- #


def parse_pdbstyle(text: str) -> tuple[str, np.ndarray]:
    """ATOM/HETATM backbone of one ASTRAL domain -> (resolved sequence, [n, 3, 3]).

    HETATM is read because ASTRAL keeps modified residues (MSE above all) as
    HETATM; ``THREE_TO_ONE`` decides whether a residue name is an amino acid.
    Residues lacking any backbone atom are dropped, and first altlocs win.
    """
    residues: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {}
    names: dict[tuple[str, str], str] = {}
    order: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE or line[16] not in (" ", "A"):
            continue
        resname = line[17:20].strip()
        if resname not in THREE_TO_ONE:
            continue
        key = (line[21], line[22:27])  # chain, resseq + insertion code
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if key not in residues:
            residues[key] = {}
            names[key] = resname
            order.append(key)
        residues[key].setdefault(atom, xyz)
    keep = [k for k in order if all(a in residues[k] for a in BACKBONE)]
    seq = "".join(THREE_TO_ONE[names[k]] for k in keep)
    coords = np.array([[residues[k][a] for a in BACKBONE] for k in keep], dtype=np.float32)
    return seq, coords.reshape(-1, 3, 3)


def place_coords(target: str, resolved: str, coords: np.ndarray) -> np.ndarray:
    """Map resolved-residue coordinates onto the full sequence; NaN where unresolved.

    ASTRAL's SEQRES-derived sequence includes residues the crystal never
    resolved. Exact matching blocks place the rest; ESM-IF1 accepts NaN
    coordinates for missing residues, which is how it was trained to see gaps.
    """
    out = np.full((len(target), 3, 3), np.nan, dtype=np.float32)
    sm = difflib.SequenceMatcher(None, target, resolved, autojunk=False)
    for blk in sm.get_matching_blocks():
        out[blk.a : blk.a + blk.size] = coords[blk.b : blk.b + blk.size]
    return out


def load_scope_structures(tarball: Path, wanted: set[str]) -> dict[str, tuple[str, np.ndarray]]:
    """One sequential pass over the ASTRAL tarball, keeping the wanted domains."""
    found: dict[str, tuple[str, np.ndarray]] = {}
    t0 = time.time()
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".ent"):
                continue
            sid = Path(member.name).stem
            if sid not in wanted:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            found[sid] = parse_pdbstyle(fh.read().decode("utf-8", errors="replace"))
            if len(found) % 2000 == 0:
                print(f"[ceiling] parsed {len(found):,} ASTRAL domains "
                      f"({time.time() - t0:.0f}s)", file=sys.stderr)
    return found


class EsmFold:
    """ESMFold via ``transformers`` -- no openfold build needed."""

    def __init__(self, device: torch.device, chunk_size: int | None = None):
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.device = device
        self.tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True)
        model.esm = model.esm.half()  # the 3B language model; the trunk stays fp32
        # Chunking trades speed for memory; at <= 512 residues the pair
        # representation fits a 24 GB card unchunked.
        model.trunk.set_chunk_size(chunk_size)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = model.eval().to(device)

    @torch.no_grad()
    def __call__(self, seq: str) -> tuple[np.ndarray, float]:
        ids = self.tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"]
        out = self.model(ids.to(self.device))
        # atom14 order begins N, CA, C for every residue type.
        coords = out["positions"][-1, 0, :, :3].float().cpu().numpy()
        plddt = float(out["plddt"][0, :, 1].float().mean())  # CA confidence
        return coords, plddt


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


class Projector:
    """The corpus PCA basis -- the exact space C3's targets live in."""

    def __init__(self, corpus: Path, device: torch.device):
        self.mean = torch.from_numpy(np.load(corpus / "pca_mean.npy")).float().to(device)
        self.comps = torch.from_numpy(np.load(corpus / "pca_components.npy")).float().to(device)
        self.scale = torch.from_numpy(np.load(corpus / "pca_scale.npy")).float().to(device)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.mean.device)
        return ((x - self.mean) @ self.comps / self.scale).half().cpu().numpy()


def build_split(
    seqs: list[str],
    structure: Callable[[int, str], tuple[np.ndarray, float | None]],
    embed: Callable[[np.ndarray], torch.Tensor],
    project: Projector,
    out_path: Path,
    max_len: int,
    chunk: int,
    stats: dict,
) -> None:
    """Embed every protein's window, saving in resumable chunks."""
    # Keyed by protein count, so a run with a different subsample never
    # resumes from another run's chunks.
    parts_dir = out_path.parent / f"_{out_path.stem}_parts_{len(seqs)}"
    parts_dir.mkdir(exist_ok=True)
    n_chunks = (len(seqs) + chunk - 1) // chunk
    t0 = time.time()
    for c in range(n_chunks):
        part = parts_dir / f"{c:05d}.npz"
        if part.exists():
            continue
        feats, plddts, resolved = [], [], []
        for i in range(c * chunk, min((c + 1) * chunk, len(seqs))):
            window = seqs[i][:max_len]
            coords, plddt = structure(i, window)
            ok = np.isfinite(coords).all(axis=(1, 2))
            resolved.append(float(ok.mean()) if len(ok) else 0.0)
            if plddt is not None:
                plddts.append(plddt)
            if not ok.any():
                # Nothing to encode. Zeros keep the flat layout aligned; the
                # count is reported so it is never silent.
                feats.append(np.zeros((len(window), project.comps.shape[1]), np.float16))
                continue
            feats.append(project(embed(coords)))
        np.savez(part, feats=np.concatenate(feats), plddt=np.asarray(plddts),
                 resolved=np.asarray(resolved))
        done = min((c + 1) * chunk, len(seqs))
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[ceiling] {out_path.name}: {done:,}/{len(seqs):,} ({rate:.1f}/s)",
              file=sys.stderr)

    feats, plddts, resolved = [], [], []
    for c in range(n_chunks):
        with np.load(parts_dir / f"{c:05d}.npz") as z:
            feats.append(z["feats"])
            plddts.append(z["plddt"])
            resolved.append(z["resolved"])
    flat = np.concatenate(feats)
    expected = sum(min(len(s), max_len) for s in seqs)
    if flat.shape[0] != expected:
        raise RuntimeError(f"{out_path.name}: {flat.shape[0]} rows, expected {expected}")
    np.save(out_path, flat)
    res = np.concatenate(resolved)
    pl = np.concatenate(plddts)
    stats.update({
        "n_proteins": len(seqs),
        "n_rows": int(flat.shape[0]),
        "mean_fraction_resolved": float(res.mean()),
        "n_without_coords": int((res == 0).sum()),
        "mean_plddt": float(pl.mean()) if pl.size else None,
    })
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-data", type=Path, default=Path("data/eval"))
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"),
                   help="build_cache output holding the PCA basis")
    p.add_argument("--tasks", nargs="*", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--scope-tarball", type=Path, default=None,
                   help=f"ASTRAL PDB-style tarball ({SCOPE_TARBALL_URL})")
    p.add_argument("--max-len", type=int, default=512, help="must match run_eval.py")
    p.add_argument("--chunk", type=int, default=250, help="proteins per resumable chunk")
    p.add_argument("--train-subsample", type=int, default=2000,
                   help="fold at most this many train proteins per task (fixed seed); "
                        "0 folds all. ESMFold runs ~0.3-1 protein/s on a 4090, so all "
                        "10.8k SS train proteins would cost ~10 GPU-hours for a "
                        "reference line")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    if "scope" in args.tasks and (args.scope_tarball is None or not args.scope_tarball.exists()):
        print(f"[ceiling] scope needs --scope-tarball; download {SCOPE_TARBALL_URL}",
              file=sys.stderr)
        return 1
    for f in ("pca_mean.npy", "pca_components.npy", "pca_scale.npy"):
        if not (args.corpus / f).exists():
            print(f"[ceiling] {args.corpus / f} missing -- run build_cache first",
                  file=sys.stderr)
            return 1

    from extract_esmif1 import encoder_embeddings, load_esmif1

    # load_esmif1 applies the fair-esm compat patches (biotite filter_backbone,
    # torch_scatter); importing esm.inverse_folding before it fails on a
    # modern stack.
    model, alphabet = load_esmif1(device)
    from esm.inverse_folding.util import CoordBatchConverter

    converter = CoordBatchConverter(alphabet)

    def embed(coords: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return encoder_embeddings(model, converter, coords, device)

    project = Projector(args.corpus, device)
    folder: EsmFold | None = None

    for task in args.tasks:
        base = args.eval_data / task
        for split in SPLITS[task]:
            out_path = base / f"{split}_esmif1.npy"
            if out_path.exists() and not args.force:
                print(f"[ceiling] {out_path} exists, skipping", file=sys.stderr)
                continue
            npz = base / f"{split}.npz"
            if not npz.exists():
                print(f"[ceiling] {npz} missing, skipping", file=sys.stderr)
                continue
            with np.load(npz) as z:
                seqs = split_sequences({k: z[k] for k in ("tokens", "offsets")})
            index = np.arange(len(seqs))
            if (split == "train" and STRUCTURE_SOURCE[task] == "esmfold"
                    and args.train_subsample and len(seqs) > args.train_subsample):
                index = np.sort(np.random.default_rng(0).choice(
                    len(seqs), args.train_subsample, replace=False))
                np.save(base / f"{split}_esmif1_index.npy", index)
                print(f"[ceiling] {task}/{split}: folding {len(index):,} of "
                      f"{len(seqs):,} proteins (seed 0)", file=sys.stderr)
            all_seqs, seqs = seqs, [seqs[i] for i in index]

            if task == "scope":
                sids = fasta_ids(base / "test.fasta")
                if len(sids) != len(all_seqs):
                    raise RuntimeError("scope test.fasta and test.npz disagree on order")
                structs = load_scope_structures(args.scope_tarball, set(sids))
                print(f"[ceiling] ASTRAL: {len(structs):,}/{len(sids):,} domains found",
                      file=sys.stderr)

                def structure(i: int, window: str, _s=structs, _ids=sids, _ix=index):
                    hit = _s.get(_ids[_ix[i]])
                    if hit is None:
                        return np.full((len(window), 3, 3), np.nan, np.float32), None
                    return place_coords(window, *hit), None
            else:
                if folder is None:
                    print("[ceiling] loading ESMFold", file=sys.stderr)
                    folder = EsmFold(device)

                def structure(i: int, window: str, _f=folder):
                    return _f(window)

            stats: dict = {}
            build_split(seqs, structure, embed, project, out_path,
                        args.max_len, args.chunk, stats)
            meta = {
                "task": task, "split": split, "max_len": args.max_len,
                "structure_source": STRUCTURE_SOURCE[task],
                "projection": "corpus PCA basis (same space as C3 targets)",
                "dim": int(project.comps.shape[1]),
                "n_proteins_in_split": len(all_seqs),
                "subsampled": bool(len(index) < len(all_seqs)),
                **stats,
            }
            if STRUCTURE_SOURCE[task] == "esmfold":
                meta["caveat"] = (
                    "structures predicted by ESMFold: this ceiling is what ESMFold + "
                    "ESM-IF1 know together, and evaluation proteins may be in "
                    "ESMFold's training data")
            (base / f"{split}_esmif1.json").write_text(json.dumps(meta, indent=2))
            print(f"[ceiling] wrote {out_path} {json.dumps(stats)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
