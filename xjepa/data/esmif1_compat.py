"""Make ``fair-esm``'s inverse-folding module importable on a modern stack.

fair-esm 2.0.0 was released against an older scientific-Python stack and has two
import-time incompatibilities with a current image. Both are trivial once
identified, and both otherwise cost GPU-billed debugging time:

1. **``torch_scatter``** -- ``gvp_modules.py`` hard-imports the compiled
   extension for two functions. Wheels are built per torch+CUDA pair, so on a
   recent image there may be none, and building from source needs nvcc and 20+
   minutes. :mod:`xjepa.data.scatter_shim` supplies pure-PyTorch equivalents.

2. **``biotite.structure.filter_backbone``** -- removed in biotite 1.0, renamed
   to ``filter_peptide_backbone``. ``util.py`` imports it at module scope, so
   the import fails before anything runs.

The tempting fix for (2) is to pin ``biotite<1.0``. Don't: biotite 0.41 requires
numpy<2, which downgrades numpy, which breaks whatever scipy the image shipped
compiled against numpy 2 (``AttributeError: module 'numpy' has no attribute
'long'``). One pin cascades into three. Aliasing the single missing symbol
leaves the rest of the environment untouched.

Note what fair-esm uses biotite *for*: reading structures out of PDB/mmCIF
files. This pipeline never does that -- ``fetch_afdb.py`` parses coordinates
itself and hands ``[L, 3, 3]`` arrays straight to the encoder. The compat layer
exists purely so a module-scope import succeeds.

Call :func:`prepare` before importing ``esm.inverse_folding``.
"""

from __future__ import annotations

import sys

__all__ = ["prepare", "report"]


def _patch_biotite() -> str:
    """Alias ``filter_backbone`` to biotite 1.x's ``filter_peptide_backbone``.

    Returns:
        ``"native"`` if biotite already provides the symbol, ``"aliased"`` if it
        was added, ``"absent"`` if biotite is not installed at all.
    """
    try:
        import biotite.structure as bs
    except ImportError:
        return "absent"

    if hasattr(bs, "filter_backbone"):
        return "native"

    replacement = getattr(bs, "filter_peptide_backbone", None)
    if replacement is None:
        return "absent"

    bs.filter_backbone = replacement  # type: ignore[attr-defined]
    # util.py does `from biotite.structure import filter_backbone`, which reads
    # the already-imported module object out of sys.modules, so patching the
    # attribute is enough -- but be explicit about it for the next reader.
    sys.modules["biotite.structure"] = bs
    return "aliased"


def prepare() -> dict[str, str]:
    """Install every compatibility shim ESM-IF1 needs on a modern stack.

    Idempotent, and each shim defers to a working native implementation.

    Returns:
        Mapping of shim name to what it did, for logging.
    """
    from xjepa.data.scatter_shim import install as install_scatter

    return {
        "torch_scatter": install_scatter(),
        "biotite.filter_backbone": _patch_biotite(),
    }


def report() -> str:
    """One-line human-readable summary of the compat state."""
    state = prepare()
    return "  ".join(f"{k}={v}" for k, v in state.items())
