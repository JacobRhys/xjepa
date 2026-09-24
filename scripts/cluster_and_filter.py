#!/usr/bin/env python3
"""Phase 2, steps 2-3: redundancy clustering and the L0 leakage filter.

Two passes over the fetched sequences, both with MMseqs2, both on CPU:

1. **Redundancy clustering** at ``--cluster-identity`` (default 50%). Keeps one
   representative per cluster, so the token budget covers more distinct
   structure instead of re-learning near-duplicates.

2. **The L0 leakage filter** (RESEARCH_PLAN.md sec. 1.6). Drops any pretraining
   sequence with >= ``--leak-identity`` identity and >= ``--leak-coverage``
   coverage to any sequence in an evaluation **test** split.

A word on what this filter does and does not buy. It does **not** address the
fact that ESM-IF1 was trained on ~12M AFDB structures spanning UniRef50, so
essentially every evaluation protein is inside its training distribution. That
leakage is not executable to remove -- filtering against "the encoder's training
set" would mean filtering against all of UniRef50 -- and the honest response is
disclosure plus the L1 ceiling probe, not a filter. What L0 removes is the
leakage that actually confounds the probe comparison: pretraining on the very
sequences the probes are later tested on.

Output is a single accession allowlist that every later stage reads, plus a
report recording exactly what was dropped and why.

Requires MMseqs2 on PATH (``brew install mmseqs2`` / ``conda install -c bioconda
mmseqs2``)::

    python scripts/cluster_and_filter.py \\
        --fasta data/structures/sequences.fasta \\
        --eval-fasta data/eval/cb513_test.fasta data/eval/scope_test.fasta \\
        --out data/splits
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def require_mmseqs() -> str:
    """Return the mmseqs binary, or exit with installation guidance."""
    exe = shutil.which("mmseqs")
    if exe is None:
        print(
            "mmseqs not found on PATH.\n"
            "  macOS:  brew install mmseqs2\n"
            "  conda:  conda install -c conda-forge -c bioconda mmseqs2\n"
            "Both passes here are CPU-only and cost nothing but minutes.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return exe


def run(cmd: list[str], label: str) -> None:
    """Run a subprocess, surfacing its stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{label}] failed:\n{proc.stderr[-4000:]}", file=sys.stderr)
        raise SystemExit(proc.returncode)


def read_fasta_ids(path: Path) -> list[str]:
    """Read sequence ids (the first whitespace-delimited token after '>')."""
    ids: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def cluster(
    mmseqs: str, fasta: Path, tmp: Path, identity: float, coverage: float, threads: int
) -> set[str]:
    """Cluster at ``identity`` and return the representative ids.

    ``easy-cluster`` writes ``<prefix>_cluster.tsv`` with one line per member,
    ``representative<TAB>member``; the representatives are the distinct values
    in column 1.
    """
    prefix = tmp / "clu"
    run(
        [
            mmseqs, "easy-cluster", str(fasta), str(prefix), str(tmp / "clu_tmp"),
            "--min-seq-id", str(identity),
            "-c", str(coverage),
            "--cov-mode", "0",
            "--threads", str(threads),
            "-v", "1",
        ],
        "cluster",
    )
    reps: set[str] = set()
    with open(f"{prefix}_cluster.tsv", "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts:
                reps.add(parts[0])
    return reps


def leaking_ids(
    mmseqs: str,
    query: Path,
    targets: list[Path],
    tmp: Path,
    identity: float,
    coverage: float,
    threads: int,
) -> dict[str, str]:
    """Find pretraining ids too similar to any evaluation test sequence.

    Returns:
        Mapping of pretraining id -> the evaluation file that matched it.
    """
    hits: dict[str, str] = {}
    for i, target in enumerate(targets):
        out = tmp / f"leak_{i}.m8"
        run(
            [
                mmseqs, "easy-search", str(query), str(target), str(out),
                str(tmp / f"leak_tmp_{i}"),
                "--min-seq-id", str(identity),
                "-c", str(coverage),
                "--cov-mode", "0",
                "-s", "7.5",              # high sensitivity: a missed hit is leakage kept
                "--max-seqs", "300",
                "--threads", str(threads),
                "--format-output", "query,target,fident,qcov",
                "-v", "1",
            ],
            f"leak-search:{target.name}",
        )
        with open(out, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                qid, fident, qcov = parts[0], float(parts[2]), float(parts[3])
                if fident >= identity and qcov >= coverage:
                    hits.setdefault(qid, target.name)
    return hits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fasta", type=Path, required=True, help="sequences.fasta from fetch_afdb.py")
    p.add_argument("--eval-fasta", type=Path, nargs="*", default=[],
                   help="evaluation TEST split FASTAs to filter against")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--cluster-identity", type=float, default=0.5)
    p.add_argument("--cluster-coverage", type=float, default=0.8)
    p.add_argument("--leak-identity", type=float, default=0.3)
    p.add_argument("--leak-coverage", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--keep-tmp", action="store_true")
    args = p.parse_args(argv)

    mmseqs = require_mmseqs()
    args.out.mkdir(parents=True, exist_ok=True)

    all_ids = read_fasta_ids(args.fasta)
    if not all_ids:
        print(f"no sequences in {args.fasta}", file=sys.stderr)
        return 1
    print(f"[filter] {len(all_ids):,} sequences in", args.fasta, file=sys.stderr)

    tmp_root = Path(tempfile.mkdtemp(prefix="xjepa_filter_"))
    try:
        print(f"[filter] clustering at {args.cluster_identity:.0%} identity ...", file=sys.stderr)
        reps = cluster(
            mmseqs, args.fasta, tmp_root,
            args.cluster_identity, args.cluster_coverage, args.threads,
        )
        print(f"[filter] {len(reps):,} cluster representatives", file=sys.stderr)

        leaks: dict[str, str] = {}
        if args.eval_fasta:
            missing = [f for f in args.eval_fasta if not f.exists()]
            if missing:
                print(f"missing eval FASTA: {missing}", file=sys.stderr)
                return 1
            print(
                f"[filter] L0 leakage search against {len(args.eval_fasta)} eval split(s) "
                f"at {args.leak_identity:.0%} identity / {args.leak_coverage:.0%} coverage ...",
                file=sys.stderr,
            )
            leaks = leaking_ids(
                mmseqs, args.fasta, list(args.eval_fasta), tmp_root,
                args.leak_identity, args.leak_coverage, args.threads,
            )
            print(f"[filter] {len(leaks):,} sequences hit an eval test split", file=sys.stderr)
        else:
            print(
                "[filter] WARNING: no --eval-fasta given, so the L0 leakage filter did NOT "
                "run. Probe results from this corpus are confounded by pretrain/test "
                "overlap and should not be reported as-is.",
                file=sys.stderr,
            )

        keep = [i for i in all_ids if i in reps and i not in leaks]
        allowlist = args.out / "pretrain_accessions.txt"
        allowlist.write_text("\n".join(keep) + "\n", encoding="utf-8")

        report = {
            "n_input": len(all_ids),
            "n_cluster_representatives": len(reps),
            "n_dropped_redundant": len(all_ids) - len(reps),
            "n_dropped_leakage": len(leaks),
            "n_kept": len(keep),
            "leakage_filter_ran": bool(args.eval_fasta),
            "params": {
                "cluster_identity": args.cluster_identity,
                "cluster_coverage": args.cluster_coverage,
                "leak_identity": args.leak_identity,
                "leak_coverage": args.leak_coverage,
            },
            "eval_splits": [f.name for f in args.eval_fasta],
            "note": (
                "L0 removes pretrain/test overlap only. It does NOT address ESM-IF1 "
                "having been trained across UniRef50, which is handled by disclosure "
                "and the L1 ceiling probe -- see RESEARCH_PLAN.md sec. 1.6."
            ),
        }
        (args.out / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if leaks:
            with open(args.out / "leaked_ids.tsv", "w", encoding="utf-8") as fh:
                fh.write("accession\tmatched_eval_split\n")
                for k, v in sorted(leaks.items()):
                    fh.write(f"{k}\t{v}\n")

        print(
            f"\n[filter] kept {len(keep):,} of {len(all_ids):,}\n"
            f"[filter]   -{len(all_ids) - len(reps):,} redundant, -{len(leaks):,} leaking\n"
            f"[filter] allowlist -> {allowlist}",
            file=sys.stderr,
        )
        return 0 if keep else 1
    finally:
        if args.keep_tmp:
            print(f"[filter] tmp kept at {tmp_root}", file=sys.stderr)
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
