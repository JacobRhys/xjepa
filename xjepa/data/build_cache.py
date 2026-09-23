"""Offline cache builder: ESM-IF1 512-d embeddings -> PCA-128 fp16 target bank.

This script runs **once**, before training, and is explicitly *not* on the
training path, so ordinary batched CPU/GPU code is fine here.  Everything is
memory-mapped: neither the 512-d source bank nor the 128-d output bank is ever
held in RAM in one piece.

Pipeline:

1. memory-map the ``[total_residues, 512]`` embedding bank;
2. fit PCA to ``--dim`` components on a random subsample (mean + covariance
   eigendecomposition, exact for the subsample);
3. project the full bank in chunks, standardising each component to unit
   variance so fp16 has headroom;
4. write ``targets.npy`` as fp16, copy ``tokens.npy`` / ``offsets.npy``
   through, and write ``meta.json``;
5. report the explained variance ratio and the RankMe of the resulting bank —
   RankMe(targets) is the ceiling line for the collapse plots (RESEARCH_PLAN
   §1.1 H1b).

Usage::

    python -m xjepa.data.build_cache \\
        --embeddings raw/esmif1_512.npy --tokens raw/tokens.npy \\
        --offsets raw/offsets.npy --out cache/ --dim 128
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch

__all__ = ["PcaFit", "fit_pca", "rankme", "transform_bank", "build_cache", "main"]


@dataclass
class PcaFit:
    """Fitted PCA transform.

    Attributes:
        mean: ``[D_in]`` float32 feature means.
        components: ``[D_in, D_out]`` float32 projection matrix (columns are
            the top eigenvectors of the covariance, descending).
        variances: ``[D_out]`` float32 eigenvalues of the kept components.
        scale: ``[D_out]`` float32 divisor applied after projection
            (component std if standardising, else ones).
        explained_variance_ratio: fraction of total variance kept.
        n_fit: number of rows the fit used.
    """

    mean: torch.Tensor
    components: torch.Tensor
    variances: torch.Tensor
    scale: torch.Tensor
    explained_variance_ratio: float
    n_fit: int


def rankme(x: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe: ``exp(entropy of L1-normalised singular values)``.

    Matches the contract for :func:`xjepa.eval.collapse.rankme`; duplicated
    here so the offline builder has no dependency on the eval package.

    Args:
        x: ``[N, D]`` matrix of representations.
        eps: floor added to the normalised spectrum for numerical stability.

    Returns:
        The effective rank, in ``[1, min(N, D)]``.
    """
    xf = x.to(torch.float32)
    sv = torch.linalg.svdvals(xf)
    p = sv / (sv.sum() + eps)
    p = p + eps
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy))


def _subsample_rows(
    bank: np.ndarray, n_sample: int, seed: int
) -> np.ndarray:
    """Draw up to ``n_sample`` random rows from a memory-mapped bank.

    Rows are fetched in sorted order so the memmap is read roughly
    sequentially.

    Args:
        bank: memory-mapped ``[N, D]`` array.
        n_sample: maximum number of rows to draw.
        seed: RNG seed.

    Returns:
        An in-RAM float32 array ``[min(n_sample, N), D]``.
    """
    n = int(bank.shape[0])
    take = min(int(n_sample), n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=take, replace=False)) if take < n else np.arange(n)
    return np.array(bank[idx], dtype=np.float32, copy=True)


def fit_pca(
    sample: np.ndarray,
    dim: int,
    standardise: bool = True,
    device: str = "cpu",
) -> PcaFit:
    """Fit PCA by eigendecomposition of the sample covariance.

    Args:
        sample: ``[n, D_in]`` float32 subsample.
        dim: number of components to keep.
        standardise: divide each projected component by its std.
        device: device to run the linear algebra on.

    Returns:
        A :class:`PcaFit`.

    Raises:
        ValueError: if ``dim`` exceeds the input width.
    """
    x = torch.as_tensor(sample, dtype=torch.float32, device=torch.device(device))
    n, d_in = x.shape
    if dim > d_in:
        raise ValueError(f"dim {dim} exceeds input width {d_in}")
    mean = x.mean(dim=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(1, n - 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0.0)
    evecs = evecs[:, order]
    comps = evecs[:, :dim].contiguous()
    kept = evals[:dim].contiguous()
    total = float(evals.sum())
    evr = float(kept.sum()) / total if total > 0 else 0.0
    scale = kept.sqrt().clamp_min(1e-6) if standardise else torch.ones_like(kept)
    return PcaFit(
        mean=mean.cpu(),
        components=comps.cpu(),
        variances=kept.cpu(),
        scale=scale.cpu(),
        explained_variance_ratio=evr,
        n_fit=int(n),
    )


def transform_bank(
    bank: np.ndarray,
    fit: PcaFit,
    out_path: str,
    chunk_rows: int = 200_000,
    device: str = "cpu",
) -> np.ndarray:
    """Project a memory-mapped bank in chunks and write fp16 to ``out_path``.

    Args:
        bank: memory-mapped ``[N, D_in]`` source embeddings.
        fit: fitted PCA transform.
        out_path: destination ``.npy`` path for the ``[N, D_out]`` fp16 bank.
        chunk_rows: rows processed per chunk.
        device: device used for the matmul.

    Returns:
        A memory-mapped handle on the written fp16 bank.
    """
    dev = torch.device(device)
    mean = fit.mean.to(dev)
    comps = fit.components.to(dev)
    scale = fit.scale.to(dev)
    n, _ = int(bank.shape[0]), int(bank.shape[1])
    d_out = int(comps.shape[1])

    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(n, d_out)
    )
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        block = torch.as_tensor(
            np.array(bank[start:stop], dtype=np.float32, copy=True), device=dev
        )
        proj = (block - mean) @ comps / scale
        out[start:stop] = proj.to(torch.float16).cpu().numpy()
    out.flush()
    return out


def _copy_npy(src: Optional[str], dst: str, chunk_rows: int = 1 << 22) -> Optional[str]:
    """Copy a ``.npy`` file through a memmap, chunk by chunk.

    Args:
        src: source path, or ``None`` to skip.
        dst: destination path.
        chunk_rows: rows copied per chunk.

    Returns:
        ``dst`` if a copy happened, else ``None``.
    """
    if src is None:
        return None
    a = np.load(src, mmap_mode="r")
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=a.dtype, shape=a.shape)
    n = int(a.shape[0])
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        out[start:stop] = a[start:stop]
    out.flush()
    return dst


def build_cache(
    embeddings: str,
    out_dir: str,
    dim: int = 128,
    tokens: Optional[str] = None,
    offsets: Optional[str] = None,
    subsample: int = 500_000,
    chunk_rows: int = 200_000,
    rankme_rows: int = 50_000,
    standardise: bool = True,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Build the fp16 PCA target bank and its metadata.

    Args:
        embeddings: path to the ``[N, 512]`` ESM-IF1 ``.npy`` bank.
        out_dir: output cache directory (created if absent).
        dim: PCA components to keep.
        tokens: optional ``tokens.npy`` to copy into the cache.
        offsets: optional ``offsets.npy`` to copy into the cache.
        subsample: rows used to fit the PCA.
        chunk_rows: rows per projection chunk.
        rankme_rows: rows used to estimate RankMe of the output bank.
        standardise: scale each component to unit variance.
        device: device for the linear algebra.
        seed: RNG seed for subsampling.

    Returns:
        The metadata dict that was written to ``meta.json``, including
        ``explained_variance_ratio`` and ``rankme``.
    """
    os.makedirs(out_dir, exist_ok=True)
    bank = np.load(embeddings, mmap_mode="r")
    if bank.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {bank.shape}")

    sample = _subsample_rows(bank, subsample, seed)
    fit = fit_pca(sample, dim=dim, standardise=standardise, device=device)

    targets_path = os.path.join(out_dir, "targets.npy")
    out = transform_bank(bank, fit, targets_path, chunk_rows=chunk_rows, device=device)

    rm_rows = min(int(rankme_rows), int(out.shape[0]))
    rng = np.random.default_rng(seed + 1)
    ridx = (
        np.sort(rng.choice(int(out.shape[0]), size=rm_rows, replace=False))
        if rm_rows < int(out.shape[0])
        else np.arange(int(out.shape[0]))
    )
    rm = rankme(torch.as_tensor(np.array(out[ridx], dtype=np.float32, copy=True)))

    _copy_npy(tokens, os.path.join(out_dir, "tokens.npy"))
    _copy_npy(offsets, os.path.join(out_dir, "offsets.npy"))

    np.save(os.path.join(out_dir, "pca_mean.npy"), fit.mean.numpy())
    np.save(os.path.join(out_dir, "pca_components.npy"), fit.components.numpy())
    np.save(os.path.join(out_dir, "pca_scale.npy"), fit.scale.numpy())

    meta = {
        "source_embeddings": os.path.abspath(embeddings),
        "total_residues": int(bank.shape[0]),
        "source_dim": int(bank.shape[1]),
        "target_dim": int(dim),
        "pca_fit_rows": fit.n_fit,
        "explained_variance_ratio": round(fit.explained_variance_ratio, 6),
        "standardised": bool(standardise),
        "rankme": round(rm, 4),
        "rankme_rows": rm_rows,
        "target_bank_bytes": int(out.shape[0]) * int(dim) * 2,
        "seed": seed,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--embeddings", required=True, help="[N, 512] .npy bank")
    ap.add_argument("--out", required=True, help="output cache directory")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--tokens", default=None, help="tokens.npy to copy through")
    ap.add_argument("--offsets", default=None, help="offsets.npy to copy through")
    ap.add_argument("--subsample", type=int, default=500_000)
    ap.add_argument("--chunk-rows", type=int, default=200_000)
    ap.add_argument("--rankme-rows", type=int, default=50_000)
    ap.add_argument("--no-standardise", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    meta = build_cache(
        embeddings=args.embeddings,
        out_dir=args.out,
        dim=args.dim,
        tokens=args.tokens,
        offsets=args.offsets,
        subsample=args.subsample,
        chunk_rows=args.chunk_rows,
        rankme_rows=args.rankme_rows,
        standardise=not args.no_standardise,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
    print(
        f"explained variance ratio (dim={meta['target_dim']}): "
        f"{meta['explained_variance_ratio']:.4f}"
    )
    print(f"RankMe(target bank) = {meta['rankme']:.2f} / {meta['target_dim']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
