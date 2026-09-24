"""The ESM-2 token alphabet, and residue encoding.

The vocabulary is ESM-2's exactly (33 tokens, `<mask>` at 32, `<pad>` at 1), so
that `xjepa.data.masking`'s PAD_ID/MASK_ID constants and the public ESM-2 t6
checkpoint used as an evaluation baseline all agree on token ids. Getting this
wrong is silent: training would work, and only the baseline comparison would be
quietly meaningless.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TOKENS",
    "TOK_TO_ID",
    "VOCAB_SIZE",
    "PAD_ID",
    "MASK_ID",
    "UNK_ID",
    "THREE_TO_ONE",
    "encode",
    "encode_many",
]

#: ESM-2 / ESM-1b alphabet in order. Index is the token id.
TOKENS: tuple[str, ...] = (
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O",
    ".", "-", "<null_1>", "<mask>",
)

TOK_TO_ID: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}
VOCAB_SIZE: int = len(TOKENS)
PAD_ID: int = TOK_TO_ID["<pad>"]
MASK_ID: int = TOK_TO_ID["<mask>"]
UNK_ID: int = TOK_TO_ID["<unk>"]

assert VOCAB_SIZE == 33 and PAD_ID == 1 and MASK_ID == 32

#: PDB three-letter residue names to one-letter codes. Non-standard residues
#: map to "X" via the encoder's fallback rather than being dropped, so residue
#: indices stay aligned with the backbone coordinate array.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "UNK": "X",
    "MSE": "M",  # selenomethionine, universally treated as MET
}


def encode(seq: str) -> np.ndarray:
    """Encode a one-letter sequence to ESM-2 token ids.

    No `<cls>` or `<eos>` is added: the corpus is stored as a flat residue array
    indexed by per-sequence offsets, so sentinel tokens would corrupt the
    alignment between tokens and per-residue structure targets.

    Args:
        seq: One-letter amino acid sequence, upper case.

    Returns:
        ``uint8`` array of length ``len(seq)``. Unknown characters become
        ``<unk>`` rather than raising, since AFDB models occasionally carry
        non-standard residues.
    """
    out = np.empty(len(seq), dtype=np.uint8)
    for i, ch in enumerate(seq):
        out[i] = TOK_TO_ID.get(ch, UNK_ID)
    return out


def encode_many(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode many sequences into the flat corpus layout.

    Args:
        seqs: One-letter sequences.

    Returns:
        ``(tokens, offsets)`` where ``tokens`` is ``uint8 [total_residues]`` and
        ``offsets`` is ``int32 [len(seqs) + 1]``, matching what
        :meth:`xjepa.data.store.GpuCorpus.load` expects on disk.
    """
    lengths = [len(s) for s in seqs]
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    if total > np.iinfo(np.int32).max:
        raise ValueError(f"corpus of {total} residues overflows int32 offsets")

    tokens = np.empty(total, dtype=np.uint8)
    for seq, start in zip(seqs, offsets[:-1]):
        tokens[start : start + len(seq)] = encode(seq)
    return tokens, offsets.astype(np.int32)
