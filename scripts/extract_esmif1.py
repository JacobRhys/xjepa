#!/usr/bin/env python3
"""Phase 3: run the frozen ESM-IF1 encoder over the backbones to build the target bank.

Produces the raw ``[total_residues, 512]`` embedding bank plus the matching
``tokens.npy`` / ``offsets.npy``, which ``xjepa.data.build_cache`` then reduces
to 128 dimensions with PCA.

Only the **encoder** of ``esm_if1_gvp4_t16_142M_UR50`` runs (Hsu et al., 2022).
It consumes backbone coordinates alone -- no sequence -- which is the entire
justification for the study: the target at a masked position carries that
residue's local geometry but not its amino-acid identity, so predicting it is
not a disguised form of masked language modelling.

Always pilot before committing the full run::

    python scripts/extract_esmif1.py --shards data/structures \\
        --allowlist data/splits/pretrain_accessions.txt \\
        --out data/raw --limit 500          # measure, then extrapolate

    python scripts/extract_esmif1.py --shards data/structures \\
        --allowlist data/splits/pretrain_accessions.txt --out data/raw

Then::

    python -m xjepa.data.build_cache --embeddings data/raw/esmif1_512.npy \\
        --tokens data/raw/tokens.npy --offsets data/raw/offsets.npy \\
        --out data/corpus --dim 128

**Install.** Verified working on a RunPod PyTorch 2.8.0 + CUDA 12.8 image::

    pip install fair-esm biotite torch_geometric
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

Run ``scripts/check_esmif1.py`` first on any new machine -- it checks the
imports and the residue alignment before an extraction is paid for.
``xjepa.data.esmif1_compat`` patches fair-esm's two import-time incompatibilities
with a modern stack, so no version pinning is needed. ``scripts/extract_3di.py``
remains the CPU-only fallback target and is worth building regardless, as a
check that results are not an ESM-IF1 artefact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xjepa.data.alphabet import encode

EMBED_DIM = 512


def load_esmif1(device: torch.device):
    """Load the frozen ESM-IF1 model, with actionable guidance if the stack is missing."""
    # fair-esm 2.0.0 has two import-time incompatibilities with a modern
    # scientific-Python stack (torch_scatter, biotite.filter_backbone). Both are
    # patched in place; see xjepa.data.esmif1_compat for why pinning biotite
    # instead cascades into a broken numpy/scipy.
    from xjepa.data.esmif1_compat import prepare

    prepare()
    try:
        import esm  # noqa: F401
        import esm.inverse_folding  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not import fair-esm inverse folding ({type(exc).__name__}: {exc}).\n"
            "ESM-IF1 needs:\n"
            "  pip install fair-esm biotite\n"
            "  pip install torch-scatter torch-sparse torch-geometric \\\n"
            "      -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html\n"
            "If this fights you, use scripts/extract_3di.py instead -- it is CPU-only "
            "and gives a discrete structural target that answers the same question.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, alphabet


def encoder_embeddings(model, batch_converter, coords: np.ndarray, device: torch.device) -> torch.Tensor:
    """Per-residue encoder output for one chain.

    ``CoordBatchConverter`` pads the coordinate array with one sentinel position
    at each end (the BOS/EOS equivalents), so the encoder returns ``L + 2`` rows
    and the real residues are ``[1:-1]``. This alignment is asserted rather than
    assumed: an off-by-one here would silently pair every residue with its
    neighbour's structure target and quietly invalidate the whole study.

    Args:
        model: The loaded ESM-IF1 model.
        batch_converter: ``esm.inverse_folding.util.CoordBatchConverter``.
        coords: ``float32 [L, 3, 3]`` N/CA/C coordinates.
        device: Where to run.

    Returns:
        ``float32 [L, 512]`` on CPU.
    """
    length = coords.shape[0]
    batch = [(coords, None, None)]
    coords_t, confidence, _, _, padding_mask = batch_converter(batch, device=device)
    out = model.encoder.forward(coords_t, padding_mask, confidence, return_all_hiddens=False)
    # [T, B, C] -> the single batch element
    rep = out["encoder_out"][0][:, 0, :]
    if rep.shape[0] != length + 2:
        raise RuntimeError(
            f"ESM-IF1 returned {rep.shape[0]} rows for a {length}-residue chain; "
            "expected L+2. The batch converter's padding convention has changed -- "
            "fix the trim before trusting any target in this bank."
        )
    return rep[1:-1].float().cpu()


def iter_chains(shard_dir: Path, allow: set[str] | None, limit: int | None) -> Iterator[tuple[str, str, np.ndarray]]:
    """Yield ``(accession, sequence, coords)`` from the fetched shards in order."""
    n = 0
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=True) as z:
            accs = list(z["accessions"])
            seqs = list(z["seqs"])
            offsets = z["offsets"]
            coords = z["coords"]
        for i, (acc, seq) in enumerate(zip(accs, seqs)):
            acc = str(acc)
            if allow is not None and acc not in allow:
                continue
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            yield acc, str(seq), coords[lo:hi]
            n += 1
            if limit is not None and n >= limit:
                return


def count_chains(shard_dir: Path, allow: set[str] | None, limit: int | None) -> tuple[int, int]:
    """Count chains and total residues in one cheap metadata pass."""
    n_chains = n_res = 0
    for shard in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=True) as z:
            accs = [str(a) for a in z["accessions"]]
            offsets = z["offsets"]
        for i, acc in enumerate(accs):
            if allow is not None and acc not in allow:
                continue
            n_chains += 1
            n_res += int(offsets[i + 1]) - int(offsets[i])
            if limit is not None and n_chains >= limit:
                return n_chains, n_res
    return n_chains, n_res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shards", type=Path, required=True, help="output dir of fetch_afdb.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allowlist", type=Path, default=None,
                   help="pretrain_accessions.txt from cluster_and_filter.py")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N chains -- use 500 to pilot the throughput first")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                   help="bank dtype on disk; fp16 halves 25 GB to 12.8 GB at 12.5M residues")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=250)
    args = p.parse_args(argv)

    allow: set[str] | None = None
    if args.allowlist:
        allow = {
            l.strip() for l in args.allowlist.read_text(encoding="utf-8").splitlines() if l.strip()
        }
        print(f"[esmif1] allowlist: {len(allow):,} accessions", file=sys.stderr)
    else:
        print(
            "[esmif1] WARNING: no --allowlist, so redundant and leaking sequences from "
            "cluster_and_filter.py are still included.",
            file=sys.stderr,
        )

    n_chains, n_res = count_chains(args.shards, allow, args.limit)
    if not n_chains:
        print(f"no chains found under {args.shards}", file=sys.stderr)
        return 1

    dtype = np.float16 if args.dtype == "float16" else np.float32
    gb = n_res * EMBED_DIM * np.dtype(dtype).itemsize / 1024**3
    print(
        f"[esmif1] {n_chains:,} chains, {n_res:,} residues -> "
        f"bank {gb:.2f} GiB as {args.dtype}",
        file=sys.stderr,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, alphabet = load_esmif1(device)

    from esm.inverse_folding.util import CoordBatchConverter

    batch_converter = CoordBatchConverter(alphabet)

    bank_path = args.out / "esmif1_512.npy"
    bank = np.lib.format.open_memmap(
        bank_path, mode="w+", dtype=dtype, shape=(n_res, EMBED_DIM)
    )

    tokens = np.empty(n_res, dtype=np.uint8)
    offsets = np.zeros(n_chains + 1, dtype=np.int64)
    accessions: list[str] = []

    cursor = 0
    n_done = 0
    n_failed = 0
    t0 = time.time()

    with torch.no_grad():
        for acc, seq, coords in iter_chains(args.shards, allow, args.limit):
            length = len(seq)
            try:
                rep = encoder_embeddings(model, batch_converter, coords, device)
            except RuntimeError as exc:
                # A hard alignment failure must stop the run; anything else
                # (OOM on one long chain, a malformed model) skips that chain.
                if "expected L+2" in str(exc):
                    raise
                n_failed += 1
                print(f"[esmif1] skipped {acc}: {exc}", file=sys.stderr)
                continue

            bank[cursor : cursor + length] = rep.numpy().astype(dtype, copy=False)
            tokens[cursor : cursor + length] = encode(seq)
            accessions.append(acc)
            cursor += length
            n_done += 1
            offsets[n_done] = cursor

            if n_done % args.log_every == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-9)
                eta = (n_chains - n_done) / max(rate, 1e-9)
                print(
                    f"[esmif1] {n_done:,}/{n_chains:,}  {rate:.1f} chains/s  "
                    f"eta {eta / 60:.1f} min",
                    file=sys.stderr,
                )

    bank.flush()
    # Chains may have been skipped, so trim to what was actually written.
    if cursor != n_res or n_done != n_chains:
        print(f"[esmif1] trimming to {n_done:,} chains / {cursor:,} residues", file=sys.stderr)
        trimmed = np.lib.format.open_memmap(
            args.out / "esmif1_512.trim.npy", mode="w+", dtype=dtype, shape=(cursor, EMBED_DIM)
        )
        step = 200_000
        for lo in range(0, cursor, step):
            trimmed[lo : lo + step] = bank[lo : lo + step]
        trimmed.flush()
        del trimmed, bank
        (args.out / "esmif1_512.trim.npy").replace(bank_path)
        tokens = tokens[:cursor]
        offsets = offsets[: n_done + 1]

    np.save(args.out / "tokens.npy", tokens)
    np.save(args.out / "offsets.npy", offsets.astype(np.int32))
    (args.out / "accessions.txt").write_text("\n".join(accessions) + "\n", encoding="utf-8")

    elapsed = time.time() - t0
    meta = {
        "n_chains": n_done,
        "n_residues": int(cursor),
        "n_failed": n_failed,
        "embed_dim": EMBED_DIM,
        "dtype": args.dtype,
        "model": "esm_if1_gvp4_t16_142M_UR50 (encoder only, frozen)",
        "licence": "CC-BY-NC 4.0 -- academic use; state this in the write-up",
        "device": str(device),
        "seconds": round(elapsed, 1),
        "chains_per_sec": round(n_done / max(elapsed, 1e-9), 2),
        "limit": args.limit,
    }
    (args.out / "extract_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\n[esmif1] {n_done:,} chains, {cursor:,} residues in {elapsed / 60:.1f} min "
        f"({meta['chains_per_sec']} chains/s)",
        file=sys.stderr,
    )
    if args.limit:
        full = 50_000
        print(
            f"[esmif1] PILOT. Extrapolated to {full:,} chains: "
            f"{full / max(meta['chains_per_sec'], 1e-9) / 3600:.2f} GPU-hours. "
            "Recompute the budget before launching the full extraction.",
            file=sys.stderr,
        )
    else:
        print(
            "[esmif1] Next: python -m xjepa.data.build_cache "
            f"--embeddings {bank_path} --tokens {args.out / 'tokens.npy'} "
            f"--offsets {args.out / 'offsets.npy'} --out data/corpus --dim 128\n"
            "[esmif1] Record the PCA explained variance AND the bank's RankMe -- "
            "that RankMe is the H1b ceiling and a go/no-go before the grid.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
