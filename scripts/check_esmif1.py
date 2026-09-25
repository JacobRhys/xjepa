#!/usr/bin/env python3
"""Verify the ESM-IF1 stack before paying for a full extraction.

Run this first on any new machine. It costs a couple of minutes and checks the
two things that can waste hours:

1. **The dependency stack imports.** ``fair-esm``'s inverse-folding module has a
   hard dependency on the compiled ``torch_scatter`` extension, whose wheels are
   built per torch+CUDA pair. When no wheel exists, building from source needs
   nvcc and 20+ minutes of billed GPU time. ``xjepa.data.scatter_shim`` provides
   pure-PyTorch equivalents of the only two functions ESM-IF1 uses, and is
   installed automatically when the real package is missing.

2. **The residue alignment.** ``CoordBatchConverter`` pads one sentinel position
   at each end, so the encoder returns ``L + 2`` rows and the real residues are
   ``[1:-1]``. ``extract_esmif1.py`` asserts this rather than assuming it,
   because an off-by-one would pair every residue with its neighbour's structure
   target -- training would proceed, losses would fall, and every downstream
   number would be quietly wrong. This script exercises that code path against
   the real model at several sequence lengths.

    python scripts/check_esmif1.py                 # full check, needs the weights
    python scripts/check_esmif1.py --imports-only  # dependency check, no download

Exit code 0 means extraction is safe to run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def ideal_helix(length: int) -> np.ndarray:
    """An idealised alpha-helix backbone: real geometry, no file I/O."""
    coords = np.zeros((length, 3, 3), dtype=np.float32)
    for i in range(length):
        theta = i * 100.0 * np.pi / 180.0
        ca = np.array([2.3 * np.cos(theta), 2.3 * np.sin(theta), 1.5 * i], dtype=np.float32)
        coords[i, 0] = ca + np.array([-1.0, 0.2, -0.5], dtype=np.float32)  # N
        coords[i, 1] = ca                                                  # CA
        coords[i, 2] = ca + np.array([1.0, -0.2, 0.5], dtype=np.float32)   # C
    return coords


def check_imports() -> int:
    """Check the dependency stack, installing the scatter shim if needed."""
    from xjepa.data.esmif1_compat import prepare
    from xjepa.data.scatter_shim import is_real_package_available

    real = is_real_package_available()
    state = prepare()
    mode = state["torch_scatter"]
    print(f"torch            {torch.__version__} (cuda {torch.version.cuda})")
    print(f"torch_scatter    {'real compiled wheel' if real else 'MISSING -> using shim'}")
    print(f"scatter backend  {mode}")
    print(f"biotite compat   {state['biotite.filter_backbone']}")

    try:
        import torch_geometric

        print(f"torch_geometric  {torch_geometric.__version__}")
    except ImportError:
        print("torch_geometric  MISSING -> pip install torch_geometric")
        return 1

    try:
        import esm

        print(f"fair-esm         {getattr(esm, '__version__', 'installed')}")
        import esm.inverse_folding  # noqa: F401
        from esm.inverse_folding.util import CoordBatchConverter  # noqa: F401

        print("inverse_folding  OK")
    except ImportError as exc:
        print(f"fair-esm         FAILED: {exc}")
        print("                 pip install fair-esm biotite")
        return 1
    return 0


def check_alignment(device: torch.device, lengths: tuple[int, ...]) -> int:
    """Run the real encoder and confirm L residues in -> L rows out."""
    from extract_esmif1 import EMBED_DIM, encoder_embeddings, load_esmif1

    print("\nloading ESM-IF1 weights (~3 GB on first run) ...", flush=True)
    model, alphabet = load_esmif1(device)
    from esm.inverse_folding.util import CoordBatchConverter

    converter = CoordBatchConverter(alphabet)
    print("weights loaded\n")

    failures = 0
    for length in lengths:
        coords = ideal_helix(length)
        try:
            with torch.no_grad():
                rep = encoder_embeddings(model, converter, coords, device)
        except RuntimeError as exc:
            print(f"L={length:<5} FAILED: {exc}")
            failures += 1
            continue

        shape_ok = rep.shape == (length, EMBED_DIM)
        finite = bool(torch.isfinite(rep).all())
        varies = float(rep.std()) > 1e-6
        status = "OK" if (shape_ok and finite and varies) else "FAIL"
        print(
            f"L={length:<5} shape={tuple(rep.shape):<12} finite={finite} "
            f"std={float(rep.std()):.4f}  {status}"
        )
        if status == "FAIL":
            failures += 1
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--imports-only", action="store_true", help="skip the weight download")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lengths", type=int, nargs="+", default=[17, 64, 131, 400])
    args = p.parse_args(argv)

    print("=== ESM-IF1 preflight ===\n")
    if check_imports() != 0:
        print("\nDEPENDENCY CHECK FAILED. Fix the above before extracting, or use "
              "scripts/extract_3di.py (Foldseek, CPU-only) as the fallback target.")
        return 1

    if args.imports_only:
        print("\nimports OK (weights not checked)")
        return 0

    failures = check_alignment(torch.device(args.device), tuple(args.lengths))
    if failures:
        print(
            f"\n{failures} ALIGNMENT FAILURE(S). Do NOT run the extraction: residues "
            "would pair with the wrong structure targets and the study would be "
            "silently invalid."
        )
        return 1

    print(
        "\nALL CHECKS PASSED. L residues in -> L rows out at every length tested, "
        "so the L+2 trim in extract_esmif1.py is correct. Extraction is safe to run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
