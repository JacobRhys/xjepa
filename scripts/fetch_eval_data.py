#!/usr/bin/env python3
"""Phase 5, step 1: fetch and convert the evaluation datasets.

Converts each benchmark into one ``.npz`` per split, in the flat layout the rest
of the repo uses, so ``scripts/run_eval.py`` never has to know about LMDB, FASTA
headers or assay CSVs.

**Read this before running.** Dataset URLs drift -- TAPE, SCOPe and ProteinGym
have all moved hosting at least once. Every source is a constant at the top of
this file and is overridable with ``--url TASK=URL``. When a download fails the
script prints the project's homepage rather than guessing a mirror, because a
silently-wrong dataset is far more expensive than a failed download. Run
``--list-sources`` to see what it will fetch and from where, and verify them
before a long run.

Anything already downloaded can be converted from disk with ``--from-local``.

    python scripts/fetch_eval_data.py --list-sources
    python scripts/fetch_eval_data.py --tasks ss scope --out data/eval
    python scripts/fetch_eval_data.py --tasks ss --from-local ~/Downloads/secondary_structure.tar.gz

Output per task, under ``--out/<task>/``:

* ``{train,valid,test}.npz`` with ``tokens`` (uint8, flat), ``offsets`` (int32)
  and task-specific label arrays
* ``test.fasta`` -- the test split's sequences, which
  ``scripts/cluster_and_filter.py`` needs for the L0 leakage filter
* ``meta.json`` -- provenance: source URL, sizes, conversion date
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
import ssl
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xjepa.data.alphabet import encode_many

USER_AGENT = "xjepa-research/0.1 (academic study)"


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS that works on a python.org macOS build (see fetch_afdb.py)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


@dataclass(frozen=True)
class Source:
    """Where one benchmark comes from and what it is for.

    ``url`` is a direct download; ``hf_repo`` is a HuggingFace dataset. TAPE's
    own S3 bucket now returns AccessDenied, so the TAPE-derived tasks come from
    mirrors -- see PROVENANCE below for what was checked before trusting them.
    """

    task: str
    url: str
    homepage: str
    note: str
    hf_repo: str = ""
    hf_files: tuple[tuple[str, str], ...] = ()  # (split, filename)
    provenance: str = ""


SOURCES: dict[str, Source] = {
    "ss": Source(
        task="ss",
        url="",
        hf_repo="proteinea/secondary_structure_prediction",
        hf_files=(("train", "training_hhblits.csv"), ("valid", "TS115.csv"),
                  ("test", "CB513.csv")),
        homepage="https://github.com/songlab-cal/tape#data",
        note="TAPE/NetSurfP-2.0 secondary structure. Train: HHblits. Valid: TS115. Test: CB513.",
        provenance=(
            "TAPE's S3 bucket returns AccessDenied, so this is the proteinea "
            "mirror. Verified before use: TS115 has exactly 115 rows as "
            "published; the column schema is NetSurfP-2.0's own (input, dssp3, "
            "dssp8, disorder, cb513_mask). CB513 has 511 rows and CASP12 20, "
            "against the 513/21 usually quoted -- a documented curation "
            "difference; report the counts actually used."
        ),
    ),
    "contact": Source(
        task="contact",
        url="",
        hf_repo="proteinea/contact_prediction",
        hf_files=(("archive", "proteinnet.tar.gz"),),
        homepage="https://github.com/songlab-cal/tape#data",
        note="TAPE contact prediction (ProteinNet). Test split: CASP12.",
        provenance=(
            "This repo hosts proteinnet.tar.gz -- the *original* TAPE archive "
            "that the dead S3 bucket served, not a re-derivation. Strongest "
            "provenance of the mirrored sets."
        ),
    ),
    "fluorescence": Source(
        task="fluorescence",
        url="",
        hf_repo="proteinea/fluorescence",
        hf_files=(("train", "fluorescence_train.csv"), ("valid", "fluorescence_valid.csv"),
                  ("test", "fluorescence_test.csv")),
        homepage="https://github.com/songlab-cal/tape#data",
        note="TAPE fluorescence landscape regression (Spearman).",
        provenance=(
            "proteinea mirror. Verified: valid split has exactly 5,362 rows as "
            "published by TAPE, and the columns carry TAPE's own field names "
            "(primary, log_fluorescence, num_mutations)."
        ),
    ),
    "stability": Source(
        task="stability",
        url="",
        hf_repo="proteinglm/stability_prediction",
        hf_files=(("train", "data/train-00000-of-00001.parquet"),
                  ("valid", "data/valid-00000-of-00001.parquet"),
                  ("test", "data/test-00000-of-00001.parquet")),
        homepage="https://github.com/songlab-cal/tape#data",
        note="TAPE stability regression (Spearman).",
        provenance=(
            "NOT a proteinea mirror -- proteinea has no stability set, so this "
            "is proteinglm's. Verified: valid split has exactly 2,512 rows as "
            "published by TAPE. Weaker provenance than the others (different "
            "publisher, reformatted to parquet with columns seq/label); state "
            "this explicitly in the write-up."
        ),
    ),
    "scope": Source(
        task="scope",
        url=(
            "https://scop.berkeley.edu/downloads/scopeseq-2.08/"
            "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
        ),
        homepage="https://scop.berkeley.edu/astral/ver=2.08",
        note="SCOPe ASTRAL 2.08 40%. Fold + superfamily come from the sccs string.",
        provenance="Fetched directly from the canonical Berkeley host.",
    ),
    "proteingym": Source(
        task="proteingym",
        url="",
        homepage="https://proteingym.org/download",
        note=(
            "ProteinGym substitution DMS assays. No stable direct URL -- download "
            "the substitutions zip from the homepage and pass --from-local."
        ),
        provenance="User-supplied download from the official site.",
    ),
}


def hf_fetch(repo: str, filename: str, cache: Path) -> tuple[Path, str]:
    """Download one file from a HuggingFace dataset, returning it and the repo SHA.

    The commit SHA is the provenance that matters: a mirror can be edited, and
    recording which revision produced these splits is what makes the run
    reproducible.
    """
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().dataset_info(repo).sha
    path = hf_hub_download(repo, filename, repo_type="dataset",
                           local_dir=str(cache / repo.replace("/", "__")))
    return Path(path), sha


def download(url: str, dest: Path, homepage: str) -> Path:
    """Download to ``dest``, with the project homepage in the failure message."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[eval] cached: {dest}", file=sys.stderr)
        return dest
    print(f"[eval] downloading {url}", file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=300, context=SSL_CONTEXT) as resp, open(dest, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        dest.unlink(missing_ok=True)
        print(
            f"\n[eval] download failed: {exc}\n"
            f"[eval] This URL may have moved. Check {homepage}, then either:\n"
            f"[eval]   --url <task>=<new url>\n"
            f"[eval]   --from-local <downloaded file>\n"
            f"[eval] Do NOT substitute a mirror you have not verified: a wrong "
            f"dataset is far more expensive than a failed download.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return dest


class EvalPusher:
    """Upload converted eval splits to HuggingFace and free the local copies.

    Phase 5's raw archives are the real disk problem, not the converted output:
    ProteinNet and the ProteinGym substitutions run to ~1 GB each, while the
    ``.npz`` splits they produce are a fraction of that. So this deletes the
    downloaded archive as soon as its conversion succeeds, and optionally the
    converted splits too once they are safely on the Hub.
    """

    def __init__(self, repo: str, prefix: str = "data/eval"):
        from huggingface_hub import HfApi

        self.repo = repo
        self.prefix = prefix.strip("/")
        self.api = HfApi()
        self.pushed = 0
        self.failed: list[str] = []
        self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    def push_dir(self, task_dir: Path, delete_local: bool = False) -> bool:
        """Upload every file in a converted task directory."""
        ok = True
        for path in sorted(task_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(task_dir.parent)
            try:
                self.api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=f"{self.prefix}/{rel.as_posix()}",
                    repo_id=self.repo,
                    repo_type="dataset",
                )
                self.pushed += 1
            except Exception as exc:  # noqa: BLE001 - never lose data to an upload
                print(f"[eval] upload FAILED for {rel}: {exc}", file=sys.stderr)
                self.failed.append(str(rel))
                ok = False
        if ok and delete_local:
            # Keep the test FASTA: cluster_and_filter.py needs it locally and it
            # is tiny compared with the splits.
            for path in sorted(task_dir.rglob("*")):
                if path.is_file() and path.name != "test.fasta":
                    path.unlink(missing_ok=True)
        return ok


def write_split(
    out_dir: Path, split: str, seqs: list[str], **arrays: np.ndarray
) -> dict[str, int]:
    """Write one split in the flat tokens/offsets layout plus its label arrays."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens, offsets = encode_many(seqs)
    np.savez(out_dir / f"{split}.npz", tokens=tokens, offsets=offsets, **arrays)
    return {"n_sequences": len(seqs), "n_residues": int(offsets[-1])}


def write_fasta(path: Path, ids: list[str], seqs: list[str]) -> None:
    """Write the test split as FASTA for the L0 leakage filter."""
    with open(path, "w", encoding="utf-8") as fh:
        for i, s in zip(ids, seqs):
            fh.write(f">{i}\n")
            for j in range(0, len(s), 60):
                fh.write(s[j : j + 60] + "\n")


# --------------------------------------------------------------------------- #
# converters
# --------------------------------------------------------------------------- #


def read_lmdb(path: Path) -> list[dict]:
    """Read a TAPE LMDB split into a list of records.

    TAPE ships its data as LMDB of pickled dicts. ``lmdb`` is an optional
    dependency precisely because this conversion runs once.
    """
    try:
        import lmdb
    except ImportError as exc:
        raise SystemExit(
            "TAPE splits are LMDB. Install the reader:  pip install lmdb\n"
            "It is needed only for this one-off conversion."
        ) from exc
    import pickle

    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    out: list[dict] = []
    with env.begin(write=False) as txn:
        n = int(txn.get(b"num_examples"))
        for i in range(n):
            out.append(pickle.loads(txn.get(str(i).encode())))
    env.close()
    return out


#: NetSurfP-2.0 label alphabets, in the order TAPE indexes them.
DSSP3 = "HEC"
DSSP8 = "GHIBESTC"


def convert_ss_csv(files: dict[str, Path], out_dir: Path) -> dict:
    """NetSurfP-2.0 CSVs -> per-residue SS3 and SS8 label arrays.

    Columns are ``input`` (sequence), ``dssp3``/``dssp8`` (same-length label
    strings) and ``cb513_mask`` (which positions are scored). Masked-out
    positions become -100 so the probe ignores them, matching how the benchmark
    is meant to be evaluated -- scoring them would inflate accuracy on residues
    the benchmark itself excludes.
    """
    import csv as _csv

    meta: dict = {"splits": {}}
    for split, path in files.items():
        with open(path, newline="") as fh:
            rows = list(_csv.DictReader(fh))
        seqs, ss3, ss8 = [], [], []
        for r in rows:
            seq = (r.get("input") or "").strip().upper()
            d3, d8 = (r.get("dssp3") or ""), (r.get("dssp8") or "")
            if not seq or len(d3) != len(seq):
                continue
            mask = r.get("cb513_mask") or ""
            keep = [
                (i < len(mask) and mask[i] not in "0 ") if mask else True
                for i in range(len(seq))
            ]
            seqs.append(seq)
            ss3.append(np.array(
                [DSSP3.index(c) if c in DSSP3 and keep[i] else -100
                 for i, c in enumerate(d3)], dtype=np.int16))
            ss8.append(np.array(
                [DSSP8.index(c) if c in DSSP8 and keep[i] else -100
                 for i, c in enumerate(d8[: len(seq)].ljust(len(seq), "C"))], dtype=np.int16))
        meta["splits"][split] = write_split(
            out_dir, split, seqs,
            ss3=np.concatenate(ss3), ss8=np.concatenate(ss8),
        )
        if split == "test":
            write_fasta(out_dir / "test.fasta", [f"ss_{i}" for i in range(len(seqs))], seqs)
    return meta


def convert_regression_table(files: dict[str, Path], out_dir: Path, task: str) -> dict:
    """Fluorescence/stability tables (CSV or parquet) -> one scalar per sequence."""
    seq_cols = ("primary", "seq", "sequence", "mutated_sequence")
    y_cols = ("log_fluorescence", "stability_score", "label", "target", "score")

    meta: dict = {"splits": {}}
    for split, path in files.items():
        if path.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(path)
            records = df.to_dict("records")
        else:
            import csv as _csv

            with open(path, newline="") as fh:
                records = list(_csv.DictReader(fh))
        if not records:
            continue
        cols = records[0].keys()
        sc = next((c for c in seq_cols if c in cols), None)
        yc = next((c for c in y_cols if c in cols), None)
        if not (sc and yc):
            raise ValueError(f"{task}/{split}: no sequence/target column in {list(cols)}")

        seqs, ys = [], []
        for r in records:
            seq = str(r[sc]).strip().upper()
            try:
                y = float(np.ravel(r[yc])[0]) if not isinstance(r[yc], str) else float(r[yc])
            except (TypeError, ValueError):
                continue
            if seq:
                seqs.append(seq)
                ys.append(y)
        meta["splits"][split] = write_split(
            out_dir, split, seqs, y=np.asarray(ys, dtype=np.float32))
        if split == "test":
            write_fasta(out_dir / "test.fasta", [f"{task}_{i}" for i in range(len(seqs))], seqs)
    return meta


def convert_tape_residue(extracted: Path, out_dir: Path) -> dict:
    """TAPE secondary structure -> per-residue SS3 and SS8 labels."""
    splits = {"train": "train", "valid": "valid", "test": "cb513"}
    meta: dict = {"splits": {}}
    for split, stem in splits.items():
        matches = list(extracted.rglob(f"*{stem}*.lmdb"))
        if not matches:
            print(f"[eval] no LMDB for split {stem}", file=sys.stderr)
            continue
        records = read_lmdb(matches[0])
        seqs = [r["primary"] for r in records]
        ss3 = np.concatenate([np.asarray(r["ss3"], dtype=np.int8) for r in records])
        ss8 = np.concatenate([np.asarray(r["ss8"], dtype=np.int8) for r in records])
        meta["splits"][split] = write_split(out_dir, split, seqs, ss3=ss3, ss8=ss8)
        if split == "test":
            write_fasta(out_dir / "test.fasta", [f"ss_{i}" for i in range(len(seqs))], seqs)
    return meta


def convert_tape_regression(extracted: Path, out_dir: Path, key: str) -> dict:
    """TAPE fluorescence/stability -> one scalar target per sequence."""
    meta: dict = {"splits": {}}
    for split, stem in {"train": "train", "valid": "valid", "test": "test"}.items():
        matches = list(extracted.rglob(f"*{stem}*.lmdb"))
        if not matches:
            continue
        records = read_lmdb(matches[0])
        seqs = [r["primary"] for r in records]
        y = np.asarray([float(np.ravel(r[key])[0]) for r in records], dtype=np.float32)
        meta["splits"][split] = write_split(out_dir, split, seqs, y=y)
        if split == "test":
            write_fasta(out_dir / "test.fasta", [f"{key}_{i}" for i in range(len(seqs))], seqs)
    return meta


def convert_scope(fasta: Path, out_dir: Path) -> dict:
    """SCOPe ASTRAL FASTA -> sequences with fold and superfamily label ids.

    ASTRAL headers look like ``>d1dlwa_ a.1.1.1 (A:) ...``; the sccs string
    ``a.1.1.1`` gives class ``a``, fold ``a.1``, superfamily ``a.1.1``. Fold
    retrieval is scored with superfamily-disjoint query/gallery splits, so both
    levels are needed.
    """
    ids: list[str] = []
    seqs: list[str] = []
    folds: list[str] = []
    supers: list[str] = []

    cur_id = cur_sccs = None
    buf: list[str] = []

    def flush() -> None:
        if cur_id and buf:
            sccs_parts = (cur_sccs or "").split(".")
            if len(sccs_parts) >= 3:
                ids.append(cur_id)
                seqs.append("".join(buf).upper())
                folds.append(".".join(sccs_parts[:2]))
                supers.append(".".join(sccs_parts[:3]))

    with open(fasta, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                buf = []
                parts = line[1:].split()
                cur_id = parts[0] if parts else None
                cur_sccs = parts[1] if len(parts) > 1 else None
            else:
                buf.append(line.strip())
    flush()

    fold_ids = {f: i for i, f in enumerate(sorted(set(folds)))}
    super_ids = {s: i for i, s in enumerate(sorted(set(supers)))}
    meta = {
        "splits": {
            "test": write_split(
                out_dir, "test", seqs,
                fold=np.asarray([fold_ids[f] for f in folds], dtype=np.int32),
                superfamily=np.asarray([super_ids[s] for s in supers], dtype=np.int32),
            )
        },
        "n_folds": len(fold_ids),
        "n_superfamilies": len(super_ids),
    }
    write_fasta(out_dir / "test.fasta", ids, seqs)
    return meta


def convert_proteingym(src: Path, out_dir: Path, max_assays: int, max_len: int) -> dict:
    """ProteinGym substitution CSVs -> a stratified subset of DMS assays.

    Each assay becomes its own ``.npz`` holding the wild-type sequence, the
    mutant sequences, the mutated positions and the measured scores.

    The scorer ranks variants by embedding distance, which is exploratory and
    expected to be weak (|rho| < 0.2): JEPA-only models expose no token
    likelihoods, so the usual pseudo-likelihood scoring is unavailable.
    """
    import csv as _csv
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    assays: list[str] = []

    def handle(name: str, text: str) -> bool:
        rows = list(_csv.DictReader(text.splitlines()))
        if not rows:
            return False
        cols = rows[0].keys()
        seq_col = next((c for c in cols if c.lower() in ("mutated_sequence", "sequence")), None)
        score_col = next((c for c in cols if c.lower() in ("dms_score", "score")), None)
        mut_col = next((c for c in cols if "mutant" in c.lower()), None)
        if not (seq_col and score_col and mut_col):
            return False

        seqs: list[str] = []
        scores: list[float] = []
        positions: list[int] = []
        for r in rows:
            mut = r[mut_col]
            if ":" in mut:  # single substitutions only
                continue
            digits = "".join(ch for ch in mut if ch.isdigit())
            if not digits:
                continue
            s = r[seq_col].strip().upper()
            if not s or len(s) > max_len:
                continue
            seqs.append(s)
            scores.append(float(r[score_col]))
            positions.append(int(digits) - 1)

        if len(seqs) < 50:
            return False
        stem = Path(name).stem
        tokens, offsets = encode_many(seqs)
        np.savez(
            out_dir / f"{stem}.npz",
            tokens=tokens,
            offsets=offsets,
            positions=np.asarray(positions, dtype=np.int32),
            scores=np.asarray(scores, dtype=np.float32),
        )
        assays.append(stem)
        return True

    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            for name in sorted(z.namelist()):
                if not name.endswith(".csv") or len(assays) >= max_assays:
                    continue
                handle(name, z.read(name).decode("utf-8", errors="replace"))
    else:
        for path in sorted(src.glob("*.csv")):
            if len(assays) >= max_assays:
                break
            handle(path.name, path.read_text(encoding="utf-8", errors="replace"))

    return {"assays": assays, "n_assays": len(assays)}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=Path("data/eval"))
    p.add_argument("--tasks", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--from-local", type=Path, default=None,
                   help="already-downloaded archive/dir for a single --tasks entry")
    p.add_argument("--url", action="append", default=[], metavar="TASK=URL",
                   help="override a source URL, repeatable")
    p.add_argument("--list-sources", action="store_true")
    p.add_argument("--max-assays", type=int, default=10, help="ProteinGym subset size")
    p.add_argument("--max-len", type=int, default=400, help="ProteinGym length cap")
    p.add_argument("--push-to", default=None, metavar="REPO",
                   help="HuggingFace dataset to upload converted splits to")
    p.add_argument("--free-space", action="store_true",
                   help="with --push-to, delete converted splits locally after upload "
                        "(test.fasta is kept -- the leakage filter needs it). Raw "
                        "archives are deleted after a successful conversion either way.")
    args = p.parse_args(argv)

    overrides = dict(u.split("=", 1) for u in args.url)

    if args.list_sources:
        print("Evaluation data sources (verify before a long run):\n")
        for s in SOURCES.values():
            print(f"  {s.task}")
            src = overrides.get(s.task) or s.url or (
                f"hf:{s.hf_repo}" if s.hf_repo else "(none -- use --from-local)")
            print(f"    source:   {src}")
            print(f"    homepage: {s.homepage}")
            print(f"    {s.note}")
            if s.provenance:
                import textwrap
                for line in textwrap.wrap(s.provenance, 74):
                    print(f"      | {line}")
            print()
        return 0

    if args.from_local and len(args.tasks) != 1:
        print("--from-local applies to exactly one --tasks entry", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "_downloads"
    pusher = EvalPusher(args.push_to) if args.push_to else None
    if pusher:
        print(f"[eval] uploading converted splits to {args.push_to}", file=sys.stderr)

    for task in args.tasks:
        src_def = SOURCES[task]
        url = overrides.get(task, src_def.url)
        out_dir = args.out / task
        print(f"\n[eval] === {task} ===", file=sys.stderr)

        hf_sha = ""
        if args.from_local:
            raw = args.from_local
            hf_files: dict[str, Path] = {}
        elif src_def.hf_repo and not overrides.get(task):
            print(f"[eval] source: {src_def.hf_repo}", file=sys.stderr)
            hf_files = {}
            for split, fn in src_def.hf_files:
                path, hf_sha = hf_fetch(src_def.hf_repo, fn, cache)
                hf_files[split] = path
            raw = hf_files.get("archive", Path())
        elif not url:
            print(
                f"[eval] {task} has no source. Download it from {src_def.homepage} "
                f"and re-run with --from-local.",
                file=sys.stderr,
            )
            continue
        else:
            hf_files = {}
            suffix = ".fa" if url.endswith(".fa") else ".tar.gz" if ".tar" in url else ".bin"
            raw = download(url, cache / f"{task}{suffix}", src_def.homepage)

        try:
            if task == "scope":
                meta = convert_scope(raw, out_dir)
            elif task == "proteingym":
                meta = convert_proteingym(raw, out_dir, args.max_assays, args.max_len)
            elif task == "ss" and hf_files:
                meta = convert_ss_csv(hf_files, out_dir)
            elif task in ("fluorescence", "stability") and hf_files:
                meta = convert_regression_table(hf_files, out_dir, task)
            else:
                extracted = cache / f"{task}_x"
                if raw.is_dir():
                    extracted = raw
                elif not extracted.exists():
                    extracted.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(raw) as tf:
                        tf.extractall(extracted, filter="data")
                if task in ("ss", "contact"):
                    meta = convert_tape_residue(extracted, out_dir)
                else:
                    meta = convert_tape_regression(
                        extracted, out_dir,
                        key="log_fluorescence" if task == "fluorescence" else "stability_score",
                    )
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(
                f"[eval] conversion failed for {task}: {type(exc).__name__}: {exc}\n"
                f"[eval] The source layout may have changed; check {src_def.homepage}.",
                file=sys.stderr,
            )
            continue

        meta.update({
            "task": task,
            "source_url": url or "local",
            "hf_repo": src_def.hf_repo or None,
            # The commit SHA is the reproducibility anchor: a mirror can be
            # edited, and this records exactly which revision produced these
            # splits.
            "hf_revision": hf_sha or None,
            "hf_files": [fn for _, fn in src_def.hf_files] or None,
            "homepage": src_def.homepage,
            "note": src_def.note,
            "provenance": src_def.provenance,
            "converted": date.today().isoformat(),
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[eval] {task} -> {out_dir}", file=sys.stderr)

        # The raw archive has served its purpose; on a constrained disk it is the
        # single biggest thing lying around (ProteinNet and ProteinGym ~1 GB each).
        if not args.from_local and isinstance(raw, Path) and raw.is_file():
            size_mb = raw.stat().st_size / 1e6
            raw.unlink(missing_ok=True)
            shutil.rmtree(cache / f"{task}_x", ignore_errors=True)
            print(f"[eval] freed {size_mb:.0f} MB of raw archive for {task}", file=sys.stderr)

        if pusher and pusher.push_dir(out_dir, delete_local=args.free_space):
            print(f"[eval] {task} uploaded", file=sys.stderr)

    print(
        f"\n[eval] done. Pass the test FASTAs to the L0 leakage filter:\n"
        f"[eval]   python scripts/cluster_and_filter.py --fasta data/structures/sequences.fasta \\\n"
        f"[eval]       --eval-fasta {args.out}/*/test.fasta --out data/splits",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
