#!/usr/bin/env python3
"""Phase 2, step 1: download AFDB structures and write backbone shards.

Fetches predicted structures from the AlphaFold Protein Structure Database
(Varadi et al., 2024), keeps the ones that pass the plan's filters, and writes
sequence plus N/CA/C backbone coordinates into shards that
``scripts/extract_esmif1.py`` consumes.

Filters (RESEARCH_PLAN.md sec. 2.1):

* length within ``--min-len`` .. ``--max-len``
* mean pLDDT >= ``--min-plddt`` (default 70). Disordered chains are dropped
  deliberately: their structure targets are noisy and near-degenerate, and
  leaving them in would depress the target bank's effective rank, which is the
  ceiling on what C3 can learn (hypothesis H1b).

``--min-len`` is a real experimental decision, not a detail. If the pilot picks
the ``crop`` bucket policy, sequences below the smallest bucket (128) have
nothing to crop into; raising the floor to 128 is clean but removes small
proteins, which are a distinct structural population. Decide it before running
this, because everything downstream is built on the output.

Runs on a laptop. Costs bandwidth and patience, not money::

    # supply your own accession list
    python scripts/fetch_afdb.py --accessions swissprot.txt --out data/structures

    # or pull reviewed UniProt accessions directly
    python scripts/fetch_afdb.py --from-uniprot --target 50000 --out data/structures

Re-running is safe: completed shards are skipped unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import re
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xjepa.data.alphabet import THREE_TO_ONE

def _ssl_context() -> "ssl.SSLContext":
    """A verified TLS context that works on a python.org macOS build.

    Those builds do not read the system keychain, so ``urlopen`` fails with
    ``CERTIFICATE_VERIFY_FAILED`` unless "Install Certificates.command" has been
    run. Pointing at certifi's bundle works everywhere without asking the user
    to run anything, and -- unlike disabling verification -- keeps the
    connections authenticated.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()

AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{ver}.pdb"
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

#: AFDB bumps its model version periodically (v4 in 2024, v6 by late 2026) and
#: retires the old file paths, so a hardcoded version silently turns every fetch
#: into a 404 -- indistinguishable from "no model for this accession". Default to
#: the current one, but discover the real version from the API the first time a
#: fetch 404s, so a future bump costs one extra request rather than a failed run.
DEFAULT_MODEL_VERSION = 6
_discovered_version: int | None = None
_discovery_lock = threading.Lock()


def discover_model_version(accession: str) -> int | None:
    """Ask the AFDB API which model version it currently serves.

    Args:
        accession: Any accession AFDB has a model for.

    Returns:
        The version integer parsed out of the API's ``pdbUrl``, or ``None`` if
        the API is unreachable or its response does not match expectations.
    """
    try:
        req = urllib.request.Request(
            AFDB_API.format(acc=accession), headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as resp:
            payload = json.loads(resp.read())
    except Exception:  # noqa: BLE001 - discovery is best effort
        return None

    entry = payload[0] if isinstance(payload, list) and payload else payload
    url = entry.get("pdbUrl", "") if isinstance(entry, dict) else ""
    match = re.search(r"model_v(\d+)\.pdb", url)
    return int(match.group(1)) if match else None
USER_AGENT = "xjepa-research/0.1 (academic study; contact via repository)"

BACKBONE_ATOMS = ("N", "CA", "C")


@dataclass
class Chain:
    """One parsed AFDB model."""

    accession: str
    seq: str
    coords: np.ndarray  # float32 [L, 3, 3] -- N, CA, C
    plddt: np.ndarray  # float32 [L]

    @property
    def mean_plddt(self) -> float:
        return float(self.plddt.mean()) if self.plddt.size else 0.0


# --------------------------------------------------------------------------- #
# accession sources
# --------------------------------------------------------------------------- #


def read_accessions(path: Path) -> list[str]:
    """Read accessions one per line, ignoring blanks and ``#`` comments."""
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out


def uniprot_accessions(target: int, min_len: int, max_len: int) -> Iterator[str]:
    """Stream reviewed (Swiss-Prot) accessions from the UniProt REST API.

    Args:
        target: Stop after yielding this many.
        min_len, max_len: Sequence length filter, applied server side so we do
            not download structures we would immediately discard.

    Yields:
        UniProt accessions.
    """
    query = f"reviewed:true AND length:[{min_len} TO {max_len}]"
    params = {"query": query, "format": "list", "size": "500"}
    url = f"{UNIPROT_SEARCH}?{urllib.parse.urlencode(params)}"
    seen = 0
    while url and seen < target:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as resp:
            body = resp.read().decode("utf-8")
            link = resp.headers.get("Link", "")
        for acc in body.split():
            yield acc
            seen += 1
            if seen >= target:
                return
        url = ""
        if 'rel="next"' in link:
            url = link[link.index("<") + 1 : link.index(">")]


# --------------------------------------------------------------------------- #
# fetching and parsing
# --------------------------------------------------------------------------- #


def fetch_pdb(accession: str, version: int, retries: int, backoff: float) -> str | None:
    """Download one AFDB model, or return ``None`` if it does not exist.

    A 404 means AFDB has no model for that accession, which is expected for a
    fraction of any accession list and is not an error. Other failures are
    retried with exponential backoff.
    """
    global _discovered_version
    with _discovery_lock:
        if _discovered_version is not None:
            version = _discovered_version
    url = AFDB_URL.format(acc=accession, ver=version)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Could be "no model for this accession", or could be that AFDB
                # has bumped its version and every path is stale. Ask once.
                with _discovery_lock:
                    if _discovered_version is None:
                        found = discover_model_version(accession)
                        _discovered_version = found if found is not None else version
                        if found is not None and found != version:
                            print(
                                f"[fetch] AFDB now serves model_v{found}, not "
                                f"v{version}; switching.",
                                file=sys.stderr,
                            )
                            version = found
                            url = AFDB_URL.format(acc=accession, ver=version)
                            continue
                return None
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
                continue
            return None
    return None


def parse_backbone(text: str, accession: str) -> Chain | None:
    """Parse N/CA/C coordinates, sequence and pLDDT out of an AFDB PDB file.

    AFDB models are single-chain, single-model, with no altlocs and with pLDDT
    in the B-factor column, so a column-slicing parser is sufficient and avoids
    a biotite/BioPython dependency for the whole of phase 2.

    Residues missing any backbone atom are dropped along with their sequence
    position, keeping ``coords`` and ``seq`` aligned index for index.

    Returns:
        The parsed :class:`Chain`, or ``None`` if nothing usable was found.
    """
    residues: dict[int, dict[str, object]] = {}
    order: list[int] = []

    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_ATOMS:
            continue
        try:
            resseq = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            bfac = float(line[60:66])
        except ValueError:
            continue
        resname = line[17:20].strip()

        rec = residues.get(resseq)
        if rec is None:
            rec = {"name": resname, "atoms": {}, "plddt": bfac}
            residues[resseq] = rec
            order.append(resseq)
        rec["atoms"][atom] = (x, y, z)  # type: ignore[index]
        if atom == "CA":
            rec["plddt"] = bfac

    seq_chars: list[str] = []
    coord_rows: list[list[tuple[float, float, float]]] = []
    plddts: list[float] = []
    for resseq in order:
        rec = residues[resseq]
        atoms = rec["atoms"]  # type: ignore[index]
        if not all(a in atoms for a in BACKBONE_ATOMS):  # type: ignore[operator]
            continue
        seq_chars.append(THREE_TO_ONE.get(str(rec["name"]), "X"))
        coord_rows.append([atoms[a] for a in BACKBONE_ATOMS])  # type: ignore[index]
        plddts.append(float(rec["plddt"]))  # type: ignore[arg-type]

    if not seq_chars:
        return None
    return Chain(
        accession=accession,
        seq="".join(seq_chars),
        coords=np.asarray(coord_rows, dtype=np.float32),
        plddt=np.asarray(plddts, dtype=np.float32),
    )


def keep(chain: Chain, min_len: int, max_len: int, min_plddt: float) -> bool:
    """Apply the plan's filters to one chain."""
    return (
        min_len <= len(chain.seq) <= max_len
        and chain.mean_plddt >= min_plddt
    )


# --------------------------------------------------------------------------- #
# sharded output
# --------------------------------------------------------------------------- #


def write_shard(path: Path, chains: list[Chain]) -> None:
    """Write one shard in the flat offset layout the rest of the pipeline uses.

    Arrays: ``accessions`` (unicode), ``seqs`` (unicode), ``offsets`` (int64,
    ``n+1``), ``coords`` (float32 ``[total_res, 3, 3]``), ``plddt`` (float32
    ``[total_res]``). Written to a temporary name and renamed, so an interrupted
    run never leaves a half-written shard that a later run would trust.
    """
    lengths = [len(c.seq) for c in chains]
    offsets = np.zeros(len(chains) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        accessions=np.array([c.accession for c in chains], dtype=object),
        seqs=np.array([c.seq for c in chains], dtype=object),
        offsets=offsets,
        coords=np.concatenate([c.coords for c in chains], axis=0),
        plddt=np.concatenate([c.plddt for c in chains], axis=0),
        allow_pickle=True,
    )
    tmp.replace(path)


class ShardPusher:
    """Upload each finished shard to a HuggingFace dataset, then delete it locally.

    Written for a machine with almost no free disk. The download itself never
    touches disk -- each PDB is parsed in memory and discarded -- but the shards
    would still accumulate to ~0.5 GB. Pushing and unlinking as we go caps peak
    local usage at a single shard (~24 MB).

    A failed upload keeps the shard on disk rather than deleting it, so a network
    blip costs bandwidth to redo rather than losing an hour of fetching.
    """

    def __init__(self, repo: str, prefix: str = "data/structures", token: str | None = None):
        from huggingface_hub import HfApi

        self.repo = repo
        self.prefix = prefix.strip("/")
        self.api = HfApi(token=token)
        self.pushed = 0
        self.failed: list[str] = []
        self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    def push(self, path: Path, delete_local: bool = True) -> bool:
        """Upload one file. Returns True on success."""
        try:
            self.api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{self.prefix}/{path.name}",
                repo_id=self.repo,
                repo_type="dataset",
            )
        except Exception as exc:  # noqa: BLE001 - never lose data to an upload error
            print(f"[fetch] upload FAILED for {path.name}: {exc}", file=sys.stderr)
            print("[fetch] keeping it on disk; re-run with --overwrite to retry",
                  file=sys.stderr)
            self.failed.append(path.name)
            return False
        self.pushed += 1
        if delete_local:
            path.unlink(missing_ok=True)
        return True


def write_fasta(path: Path, chains: Iterable[Chain]) -> int:
    """Append chains to a FASTA, for MMseqs2 in the next step."""
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for c in chains:
            fh.write(f">{c.accession}\n")
            for i in range(0, len(c.seq), 60):
                fh.write(c.seq[i : i + 60] + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--accessions", type=Path, help="file of UniProt accessions, one per line")
    src.add_argument("--from-uniprot", action="store_true", help="stream reviewed accessions")

    p.add_argument("--out", type=Path, required=True, help="output directory for shards")
    p.add_argument("--target", type=int, default=50_000, help="how many chains to keep")
    p.add_argument("--min-len", type=int, default=40,
                   help="raise to 128 if the pilot chose the crop bucket policy")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--min-plddt", type=float, default=70.0)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent downloads; be polite to a public service")
    p.add_argument("--model-version", type=int, default=DEFAULT_MODEL_VERSION,
                   help="AFDB model version; auto-corrected from their API on a 404")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--backoff", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--push-to", default=None, metavar="REPO",
                   help="HuggingFace dataset to stream shards into (e.g. user/xjepa). "
                        "Each shard is uploaded then deleted locally, capping peak "
                        "disk use at one shard -- for machines short on space.")
    p.add_argument("--keep-local", action="store_true",
                   help="with --push-to, upload but do not delete the local shard")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    pusher = ShardPusher(args.push_to) if args.push_to else None
    if pusher:
        print(f"[fetch] streaming shards to {args.push_to} (peak local disk: one shard)",
              file=sys.stderr)
    fasta = args.out / "sequences.fasta"
    if args.overwrite and fasta.exists():
        fasta.unlink()

    if args.accessions:
        accessions: Iterable[str] = read_accessions(args.accessions)
    else:
        print("[fetch] streaming reviewed accessions from UniProt ...", file=sys.stderr)
        accessions = uniprot_accessions(
            target=args.target * 3,  # oversample: not every accession has a model
            min_len=args.min_len,
            max_len=args.max_len,
        )

    kept: list[Chain] = []
    shard_idx = 0
    n_kept = n_seen = n_missing = n_filtered = 0
    t0 = time.time()

    def work(acc: str) -> Chain | None:
        text = fetch_pdb(acc, args.model_version, args.retries, args.backoff)
        if text is None:
            return None
        return parse_backbone(text, acc)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for chain in pool.map(work, accessions):
            n_seen += 1
            if chain is None:
                n_missing += 1
            elif not keep(chain, args.min_len, args.max_len, args.min_plddt):
                n_filtered += 1
            else:
                kept.append(chain)
                n_kept += 1

            if len(kept) >= args.shard_size:
                path = args.out / f"shard_{shard_idx:04d}.npz"
                if args.overwrite or not path.exists():
                    write_shard(path, kept)
                    write_fasta(fasta, kept)
                    if pusher:
                        pusher.push(path, delete_local=not args.keep_local)
                shard_idx += 1
                kept = []

            if n_seen % 500 == 0:
                rate = n_seen / max(time.time() - t0, 1e-9)
                print(
                    f"[fetch] seen {n_seen:,}  kept {n_kept:,}  no-model {n_missing:,}  "
                    f"filtered {n_filtered:,}  ({rate:.1f}/s)",
                    file=sys.stderr,
                )
            if n_kept >= args.target:
                break

    if kept:
        path = args.out / f"shard_{shard_idx:04d}.npz"
        write_shard(path, kept)
        write_fasta(fasta, kept)
        if pusher:
            pusher.push(path, delete_local=not args.keep_local)
        shard_idx += 1

    meta = {
        "n_kept": n_kept,
        "n_seen": n_seen,
        "n_no_model": n_missing,
        "n_filtered_out": n_filtered,
        "n_shards": shard_idx,
        "filters": {
            "min_len": args.min_len,
            "max_len": args.max_len,
            "min_plddt": args.min_plddt,
        },
        "afdb_model_version": args.model_version,
        "source": "uniprot-reviewed" if args.from_uniprot else str(args.accessions),
    }
    if pusher:
        meta["pushed_shards"] = pusher.pushed
        meta["failed_uploads"] = pusher.failed
    (args.out / "fetch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if pusher:
        # The FASTA and metadata are small and needed by the next step.
        pusher.push(fasta, delete_local=False)
        pusher.push(args.out / "fetch_meta.json", delete_local=False)
        print(f"[fetch] uploaded {pusher.pushed} files to {args.push_to}", file=sys.stderr)
        if pusher.failed:
            print(f"[fetch] {len(pusher.failed)} uploads FAILED: {pusher.failed}",
                  file=sys.stderr)

    print(
        f"\n[fetch] kept {n_kept:,} chains in {shard_idx} shards -> {args.out}\n"
        f"[fetch] {n_missing:,} accessions had no AFDB model; "
        f"{n_filtered:,} failed the length/pLDDT filters\n"
        f"[fetch] FASTA for clustering: {fasta}",
        file=sys.stderr,
    )
    return 0 if n_kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
