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
pytest                                            # CPU, tiny shapes

# Phase 1 -- pilot. Gates everything else: settles bucket policy and head
# count by measurement, and recomputes the budget from measured throughput.
python scripts/pilot.py --out runs/pilot

# Phase 2 -- data (laptop, free)
python scripts/fetch_afdb.py --from-uniprot --target 50000 --out data/structures
python scripts/fetch_eval_data.py --list-sources     # verify URLs before trusting them
python scripts/fetch_eval_data.py --out data/eval
python scripts/cluster_and_filter.py --fasta data/structures/sequences.fasta \
    --eval-fasta data/eval/*/test.fasta --out data/splits

# Phase 3 -- structure targets. Pilot the throughput before the full run.
python scripts/extract_esmif1.py --shards data/structures \
    --allowlist data/splits/pretrain_accessions.txt --out data/raw --limit 500
python -m xjepa.data.build_cache --embeddings data/raw/esmif1_512.npy \
    --tokens data/raw/tokens.npy --offsets data/raw/offsets.npy \
    --out data/corpus --dim 128
python scripts/extract_3di.py --shards data/structures \
    --allowlist data/splits/pretrain_accessions.txt --out data/raw_3di

# Phase 4 -- the grid, in tiers, under a hard spend cap
python scripts/run_grid.py --corpus data/corpus --out runs/ --tier 1
python scripts/run_grid.py --corpus data/corpus --out runs/ --tier 2 --tier 3

# Phases 5-6 -- evaluate and aggregate
python scripts/run_eval.py --runs runs/ --eval-data data/eval --out results/
python scripts/aggregate_results.py --results results/ --out report/
```

### Running it on a rented pod

`scripts/pod_session.sh` runs one phase and then terminates the instance. A
forgotten pod costs roughly £6 a night -- the whole budget -- and monitoring does
not protect against that; the instance ending itself does.

```bash
export HF_REPO=youruser/xjepa HF_TOKEN=hf_...   # results land here
export RUNPOD_API_KEY=...                        # RUNPOD_POD_ID is already set

./scripts/pod_session.sh pilot --no-terminate    # first time: watch it
./scripts/pod_session.sh extract                 # needs a 60 GB container disk
./scripts/pod_session.sh grid --tier 1
./scripts/pod_session.sh eval
```

It refuses to start without somewhere to put results, runs the test suite before
spending GPU time on a broken tree, pushes a heartbeat every 10 minutes so
progress is visible without SSH, and enforces a hard wall-clock ceiling per
phase. **If the results push fails it does not terminate** -- another hour of
credit is recoverable, four hours of lost extraction is not.

### Three gates

Stop and look at the numbers before spending more:

1. **Pilot invariants** -- host-to-device bytes and implicit syncs must be zero.
   If not, the efficiency design has a hole.
2. **Target bank RankMe** (from `build_cache`'s `meta.json`) -- this is the H1b
   ceiling on what C3 can learn. A low value is a go/no-go, and it is known
   before any grid time is spent.
3. **Tier 1 collapse check** -- `run_grid.py` prints it. If C2 does not collapse,
   the replication failed and the framing needs rethinking; tier 1 is the
   cheapest place to find that out.
