# Shared interfaces — do not change without updating every consumer

All modules import from these signatures. Agents own disjoint directories.

## Core efficiency invariant

**The entire dataset lives in VRAM. There are NO host->device transfers inside the training loop.**

At our scale this is not an optimisation, it is a structural property:

| Asset | Size | Resident |
|---|---|---|
| Token ids, 12.5M residues, uint8 | 11.9 MB | GPU |
| Sequence offsets + lengths, int32 | 0.4 MB | GPU |
| Target bank, 12.5M x 128, fp16 (PCA-reduced) | 2.98 GiB | GPU |
| Model + optimiser states, 8M params | ~0.2 GB | GPU |

Total ~2.99 GiB on a 24 GB card (3.6 GiB if the corpus reaches 15M residues).
This is AT the 3 GB line, not comfortably under it -- `GpuCorpus.summary()` reports
the real figure at startup, so size from that, never from this table. There is therefore **no DataLoader, no worker processes,
no pin_memory, no prefetch stream, and no collate function.** Batching is index arithmetic
on device. Any PR that introduces a `torch.utils.data.DataLoader` in the training path is wrong.

## Contracts

### xjepa/data/store.py
```python
class GpuCorpus:
    """Whole corpus resident on device. Nothing is transferred per step."""
    tokens:  torch.Tensor  # uint8   [total_residues]      device
    offsets: torch.Tensor  # int32   [n_seqs + 1]          device
    targets: torch.Tensor  # float16 [total_residues, 128] device
    lengths: torch.Tensor  # int32   [n_seqs]              device

    @classmethod
    def load(cls, path: str, device: str, target_dim: int) -> "GpuCorpus": ...

class BucketBatcher:
    """Yields fixed-shape batches so torch.compile caches a small set of graphs.

    Buckets: 128 / 256 / 384 / 512. Batch size per bucket is chosen to hold
    tokens-per-step ~constant (token-budget batching), so padding waste stays
    under 8% instead of the ~48% a pad-to-512 collate would burn.
    All index math runs on device; the only host->device traffic is the step counter.
    """
    def __iter__(self) -> Iterator["Batch"]: ...

@dataclass
class Batch:
    tokens:  torch.Tensor  # int64   [B, L]   device
    targets: torch.Tensor  # float16 [B, L, 128]
    pad_mask: torch.Tensor # bool    [B, L]   True = real residue
    mask_sel: torch.Tensor # bool    [B, L]   True = masked position
    labels:  torch.Tensor  # int64   [B, L]   -100 where not masked
    bucket:  int
```

### xjepa/model/encoder.py
```python
@dataclass
class EncoderConfig:
    n_layers: int = 6; d_model: int = 320; n_heads: int = 20
    d_ff: int = 1280; vocab: int = 33; max_len: int = 512; rope: bool = True

class Encoder(nn.Module):
    def forward(self, tokens, pad_mask) -> torch.Tensor:  # [B, L, d_model]
        ...
```

### xjepa/model/heads.py
```python
class Predictor(nn.Module):   # 2 layers, d=160, 4 heads -> narrow, per I-JEPA
    def forward(self, h, mask_sel, pad_mask=None) -> torch.Tensor: ...  # [B, L, 128]
    # pad_mask is keyword-optional but objectives MUST pass it, or the predictor
    # attends over padding. Output is dense [B, L, 128]; the objective selects
    # positions. replace_masked=False gives c5c_predictor_nomask directly.
class MlmHead(nn.Module):
    def forward(self, h) -> torch.Tensor: ...             # [B, L, vocab]
```

### xjepa/train/objectives.py
```python
class Objective(Protocol):
    name: str
    def loss(self, model, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (scalar loss, metrics). Metrics values MUST stay as GPU
        tensors -- never call .item() here; the trainer syncs on its own cadence."""
```
Conditions: `c1_mlm`, `c2_jepa_ema`, `c3_jepa_frozen`, `c4_mlm_jepa`,
`c5_distil`, `c5b_masked_distil`, `c5c_predictor_nomask`.

### Labels carry clean ids at ALL masked positions

`Batch.labels` holds the **original** token id at every masked position --
including the ESM-2 10%-keep and 10%-random branches -- and `-100` at padding
and unmasked positions. `c5_distil` and `c5c_predictor_nomask` reconstruct the
unmasked sequence as `where(labels >= 0, labels, tokens)`; a label set only at
the 80% `[MASK]` branch would silently feed them corrupted tokens.

### The predictor must never overwrite every position

`c5c_predictor_nomask` predicts at *all* positions, so its predictor is built
with `replace_masked=False`. Building it with the default `True` and feeding
`pad_mask` in as `mask_sel` substitutes the learned mask query at every real
residue, discards the encoder output entirely, and **severs the gradient path
to the encoder** -- the run trains nothing and the loss still looks healthy.
`Objective.predictor_replaces_masked` carries this flag; c5c is the only
condition that sets it `False`. Caught by `tests/test_integration.py`, which
asserts non-zero encoder gradients for every condition.

### Dtype rule at the objective boundary

`Batch.targets` is fp16. Encoder/predictor output under bf16 autocast is **fp32**
(LayerNorm sits on the autocast fp32 list). Objectives must cast explicitly --
`targets.to(pred.dtype)` -- never rely on type promotion.

### Actual parameter counts (measured, not nominal)

ESM-2 t6's "8M" is a round-up. The faithful trunk is **7,408,960** parameters;
+ tied MLM head 103,393 (C1 = 7,512,353), + predictor 691,008 (C2/C3 = 8,099,968).
Do **not** widen `d_ff` to chase a round 8M -- fidelity to the reference config is
worth more than the round number, and any architecture change confounds C1-C5.
Use **N = 7.41e6** in every FLOP/MFU calculation, not 8e6.

### xjepa/eval/collapse.py
```python
def rankme(x: torch.Tensor) -> float:      # exp(entropy of normalised singular values)
def dim_std(x: torch.Tensor) -> float:
def offdiag_cov_mass(x: torch.Tensor) -> float:
```

## Hard rules for every agent

1. **No `.item()`, `.cpu()`, `.numpy()`, `float()` or `print()` of a tensor inside the
   training step.** Each forces a device sync and stalls the pipeline. Accumulate metrics
   into a preallocated GPU tensor; the trainer syncs once every `log_every` steps.
2. **No `torch.cuda.synchronize()`** outside the benchmark harness.
3. Fixed shapes per bucket. No `dynamic=True` on `torch.compile`.
4. `fused=True` on AdamW. **Not** `foreach=True` as well -- torch raises
   ``"`fused` and `foreach` cannot be `True` together."`` Fused is already a
   multi-tensor kernel; fall back to `foreach=True` only where fused is
   unavailable (CPU, older CUDA).
4b. **An "epoch" does not mean every residue once.** Hitting the padding target
   requires cropping long sequences down to the next bucket (default `hybrid`
   policy: pad up when bucket occupancy >= 0.85, else random-crop down).
   Measured: 95.7% padding efficiency with 14.2% of residues cropped per epoch.
   The crop window is redrawn on device every epoch, so coverage is reached
   across epochs, not within one. The fixed budget is **tokens**, not epochs --
   report tokens seen, never epochs. Identical policy across all conditions,
   so it is controlled; it must still be stated in the methods.
5. Attention via `F.scaled_dot_product_attention` only.
6. bf16 autocast; fp32 master weights.
7. Every module gets a test in `tests/` that runs on CPU with tiny shapes.
