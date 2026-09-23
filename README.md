# xjepa — cross-modal JEPA for protein sequences

Does latent prediction learn useful protein sequence representations without MLM, when a
**frozen structure encoder** supplies the targets?

Ofer et al. (2026) found JEPA-only training on protein sequences collapses almost everywhere,
and helps only alongside MLM. In their setup the target encoder is learned jointly with the
context encoder, which admits the trivial solution of a constant embedding. This repo removes
that trivial solution by taking targets from a frozen, pretrained ESM-IF1 structure encoder.

Full protocol, costing and pre-registered predictions: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).
Shared interfaces and performance rules: [`docs/CONTRACTS.md`](docs/CONTRACTS.md).
Performance rationale: [`docs/PERF.md`](docs/PERF.md).

## Conditions

| id | objective | role |
|---|---|---|
| `c1_mlm` | masked amino acid prediction | baseline |
| `c2_jepa_ema` | latent prediction, EMA target | collapse replication |
| `c3_jepa_frozen` | latent prediction, frozen ESM-IF1 target | the treatment |
| `c4_mlm_jepa` | MLM + λ·JEPA | complementarity |
| `c5_distil` | regress all positions, no mask, no predictor | distillation control |
| `c5b_masked_distil` | masked positions, no predictor | isolates the predictor |
| `c5c_predictor_nomask` | all positions, with predictor | isolates the masking |

Config files differ **only** in their objective block. That diffability is the experimental control.

## Design constraint: the corpus lives in VRAM

The whole training corpus is under 3 GB:

| asset | size |
|---|---|
| tokens, 10M residues, uint8 | 10 MB |
| offsets, int32 | 0.3 MB |
| ESM-IF1 target bank, 10M × 128, fp16 | 2.6 GB |
| model + optimiser, 8M params | ~0.2 GB |

So it is loaded to device once and never transferred again. There is no `DataLoader`, no worker
process, no `pin_memory`, no prefetch stream and no collate function anywhere in the training
path — batching is index arithmetic on device. At 8M parameters and length 512 the bottleneck is
kernel launch overhead and attention efficiency, not bandwidth, and the code is written to keep
it that way: no implicit device syncs in the hot loop, fixed shapes per bucket so `torch.compile`
caches a handful of graphs, fused optimiser, SDPA attention.

## Budget

Whole study — 23 runs, 3 seeds on the primary conditions — costs about **£6** of rented
RTX 4090 time. See `RESEARCH_PLAN.md` §6.

## Layout

```
xjepa/data/    GPU-resident corpus, bucket batching, masking, target cache build
xjepa/model/   ESM-2 t6 style encoder (8M), RoPE, narrow predictor, MLM head, EMA
xjepa/train/   seven objectives, trainer, schedule, checkpointing
xjepa/eval/    collapse diagnostics (RankMe, VICReg variance), probes, retrieval
xjepa/perf/    benchmark and profiling harness
```

## Running

```bash
pip install -e ".[dev]"
pytest                                    # CPU, tiny shapes
python -m xjepa.perf.bench                # throughput, MFU, per-step transfer bytes
python -m xjepa.train.trainer --config configs/c3_jepa_frozen.yaml --seed 0
```
