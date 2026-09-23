# Performance: what we optimise, what we measure, and what actually limits us

Scope: the pretraining step for the 5-condition grid (RESEARCH_PLAN.md §3–4).
Model: ESM-2 t6 shape — 6 layers, d_model 320, 20 heads, FFN 1280, RoPE, L = 512.
Target hardware: one RTX 4090 (24 GB, dense bf16 peak **165.2 TFLOP/s**).

Everything below is produced by `xjepa/perf/bench.py` and `xjepa/perf/profile_report.py`:

```bash
python -m xjepa.perf.bench --steps 50 --warmup 10 --batch-size 128 --seq-len 512 \
    --json out/bench.json --markdown out/bench.md
python -m xjepa.perf.profile_report --steps 8 --out docs/profile_latest.md
```

Rows marked **measured** come from a run on the target card. Rows marked *modelled*
are arithmetic, stated so a reviewer can falsify them with one command. At the time
of writing this repo has been developed on a machine with no CUDA device, so the GPU
columns are *modelled* and the harness has been verified end-to-end on CPU only.
Do not quote the modelled numbers as results.

---

## 1. Parameter counts — get these right before computing anything

"8M" is a round-up of the ESM-2 t6 config and using it inflates MFU by ~8%.

| Quantity | Params | Used as N for |
|---|---|---|
| Encoder alone | **7,408,960** | C1 baseline, and the honest cross-condition denominator |
| + tied MLM head | 7,512,353 | C1 as actually run |
| + predictor (2 layers, d 160) | 8,099,968 | C2 / C3 / C4 / C5c |

The JEPA conditions carry ~0.69M extra predictor parameters — a 9.3% spread. **Any
MFU table comparing conditions must state which N it used**, otherwise C3 looks 9%
more efficient than C1 for no reason other than arithmetic. `bench.py` reports the N
it used on every line (`--params-basis {measured,encoder,c1_mlm_tied,c2_c3_jepa}`)
and defaults to the built model's measured non-embedding count.

FLOP accounting (`transformer_flops_per_step`):

```
FLOPs/token = 6 * N_nonembed  +  12 * n_layers * d_model * seq_len
```

At N = 7.41M, L = 512: 44.5 MFLOP/token from parameters + 11.8 MFLOP/token from
attention = 56.3 MFLOP/token. **The attention term is 21% of the total** — not the
rounding error that "6ND is close enough" assumes. At the planned 128×512 = 65,536
tokens/step that is **3.69 TFLOP/step**, and 5.63e16 FLOP for a 1B-token run.

---

## 2. Why this workload has no transfer bottleneck

### 2.1 The whole corpus is resident

| Asset | Size | Where |
|---|---|---|
| Token ids, 10M residues, uint8 | 10 MB | VRAM |
| Sequence offsets, int32 | 0.3 MB | VRAM |
| Target bank, 10M × 128, fp16 (PCA-reduced) | 2.56 GB | VRAM |
| Model + AdamW states, 7.4M params fp32 | ~0.12 GB | VRAM |
| Activations, batch 128 × 512, bf16 | ~1–2 GB | VRAM |

Under 5 GB on a 24 GB card. There is therefore **no `DataLoader`, no worker
processes, no `pin_memory`, no prefetch stream and no collate function**. A batch is
`index_select` on tensors that are already on the device. Per-step host→device
traffic is **0 bytes**, and `bench.py` proves it by counting kineto `Memcpy HtoD`
events rather than asserting it.

### 2.2 The honest version: bandwidth was never the threat

It would be overselling to claim we dodged a bandwidth wall. Do the arithmetic for a
naive loader that streams each batch from host memory:

* targets: 128 × 512 × 128 × 2 B = **16.8 MB/step**
* tokens: 128 × 512 × 8 B = 0.5 MB/step
* at a modelled ~90 ms/step that is **~190 MB/s**, about 1% of PCIe 4.0 ×16.

So a DataLoader would not saturate the bus. What it *would* cost is everything else:

1. **Synchronisation points.** Every `.item()`, `.cpu()` or Python-side indexing of a
   device tensor stalls the pipeline for the full queue depth. Contract hard rule 1
   exists for this; `bench.py` counts violations with
   `torch.cuda.set_sync_debug_mode("warn")`.
2. **Copy-on-the-critical-path.** A non-pinned, non-overlapped H2D copy is
   synchronous with respect to the compute stream; 16.8 MB of pageable memory costs
   ~3–5 ms including the staging copy — 4–5% of the step, for nothing.
3. **Worker jitter and startup.** 23 short runs × worker spin-up, plus tail latency
   whenever a worker misses its slot.
4. **Host RAM and CPU contention** on a cheap community-cloud pod with few vCPUs.

The correct claim is therefore: *at this scale the transfer volume is small enough
that keeping the corpus resident is free, so we take the structural guarantee of
zero copies and zero syncs rather than managing a pipeline we do not need.* The
efficiency win is latency and determinism, not bandwidth.

### 2.3 What is still allowed to cross the bus

Exactly two things, neither per-step-critical:

* the **step counter** and RNG seeding (host → kernel launch arguments, not a memcpy);
* **logging**, every `log_every` steps, when accumulated metric tensors are synced
  once. This is a deliberate D2H of a few hundred bytes on a fixed cadence.

The collapse monitor (`xjepa/eval/collapse.py`) also runs on a fixed cadence and on
a *pre-allocated, device-resident* probe batch; its selection indices are computed
once at construction, so it neither transfers nor grows.

---

## 3. What the real bottleneck is at 8M params and L = 512

Short version: **attention kernel efficiency, then GEMM shape, then launch
overhead** — and bandwidth nowhere.

### 3.1 Attention does not get the flash kernel

We pass an arbitrary key-padding mask to `F.scaled_dot_product_attention`. PyTorch's
flash backend accepts only `is_causal` or no mask at all, so dispatch lands on
**mem-efficient attention (cutlassF)**. `bench.py` reports the predicted backend
(`detect_sdpa_backend`) *and* the attention kernel names the profiler actually
recorded, so this is measured, not assumed.

Worse, `head_dim = 320 / 20 = 16`. Attention kernels are tuned for head dims of
32/64/128; at 16 the tensor-core tile is more than half empty. Modelled:

| Component | FLOPs/step | Modelled kernel efficiency | Modelled time |
|---|---|---|---|
| GEMMs (QKV, proj, FFN) | 2.92 TFLOP | ~45% of peak | ~39 ms |
| Attention (scores + context) | 0.77 TFLOP | ~8% of peak (cutlassF, head_dim 16) | ~58 ms |
| LayerNorm / GELU / residual (memory-bound) | — | ~3–6 GB traffic @ ~1 TB/s | ~4–6 ms |
| Kernel launch gaps | — | ~300–600 launches × 3–5 µs | ~2–3 ms |
| Optimiser (fused AdamW) | — | bandwidth-bound on 7.4M params | ~1 ms |

**Attention is 21% of the FLOPs and a modelled ~55% of the time.** That is the
number to attack first. If a flash-eligible path were available the same attention
would plausibly run at 15–20% of peak (~25–30 ms) and the step would drop from
~100 ms to ~70 ms — a **~1.4× end-to-end speedup**, worth roughly £1.5 of the £6
budget across the grid.

Options, in order of preference:

1. **Length-homogeneous buckets with zero intra-bucket padding.** If every sequence
   in a batch is exactly the bucket length, no mask is needed and flash is eligible.
   Costs a little batching flexibility; the token-budget batcher is already close to
   this.
2. **Accept cutlassF** and report it. Defensible, and it is what the modelled numbers
   above assume.
3. Do *not* "fix" it by dropping the mask and letting real residues attend to
   padding. That silently changes the objective.

### 3.2 GEMM shapes are adequate, not good

The FFN matmul is M = 65,536, K = 320, N = 1,280. Arithmetic intensity ≈ 254
FLOP/byte against a 4090 ridge point of ~164, so it is compute-bound — good. But
K = 320 is small: each output tile does only 320 MACs before writing out, so the
reduction is short relative to the epilogue and the k-loop never reaches steady
state. Expect ~40–50% of peak on these, not the ~70% a larger model gets.

### 3.3 Kernel launch overhead is real but secondary — at this batch size

At 65,536 tokens/step the step is ~100 ms and launch overhead is ~2–3% of it. Launch
overhead becomes dominant only when tokens-per-step falls. Two things protect us:

* **token-budget batching** keeps tokens/step ~constant across the 128/256/384/512
  buckets, so the small-bucket steps are not launch-bound either;
* **`torch.compile` with fixed shapes** fuses the elementwise chains, cutting both
  the launch count and the memory traffic in §3.1 row 3.

If measured MFU comes in well below the model above, check the launch count in the
profile report *before* assuming the kernels are slow — a recompile storm or a graph
break puts eager-mode Python back in the inner loop and looks exactly like this.

### 3.4 Realistic MFU target

| Scenario | Modelled MFU | Modelled 1B-token run |
|---|---|---|
| Eager, no compile | 10–14% | 40–57 min |
| `torch.compile`, cutlassF attention (**expected**) | **18–25%** | 23–32 min |
| `torch.compile` + flash-eligible attention | 28–33% | 17–20 min |
| Hard ceiling at this shape | ~40% | ~14 min |

RESEARCH_PLAN.md §3.4 assumed 25% MFU and ~20 min/run. That sits at the **optimistic
edge** of the expected band. The plan's own gate applies: run the 200-step pilot,
recompute the budget from measured throughput, and if MFU < 15% drop to 500M
tokens/run and say so. At 18% MFU the 23-run grid is ~12 GPU-hours ≈ $4.10 — still
inside budget, but the contingency is thinner than the plan's 7.7 h assumes.

A note on honesty in the write-up: **MFU at 7.4M parameters is not a measure of
engineering quality.** Small models are latency- and shape-bound; 20% MFU here is a
good result, whereas 20% on a 7B model would be a bug. Report the number with the
model size next to it.

---

## 4. What the harness measures, and how

| Quantity | Method | Failure mode it catches |
|---|---|---|
| steps/sec, tokens/sec | wall clock over the measured window, warmup discarded | everything |
| fwd / bwd / opt split | `torch.cuda.Event` pairs, one `synchronize()` after the loop | a slow optimiser, an unexpectedly heavy backward |
| MFU | measured TFLOP/s ÷ card dense bf16 peak (table in `bench.py`) | silent throughput regressions |
| H2D / D2H bytes per step | kineto `Memcpy` events from the chrome trace (`args.bytes`), plus an nvtx range per step for `nsys` | a tensor being built on the host inside the loop |
| implicit device syncs | `torch.cuda.set_sync_debug_mode("warn")` + captured warnings | `.item()` / `.cpu()` / `float()` in the step (hard rule 1) |
| attention backend | `torch.backends.cuda.SDPAParams` + observed kernel names | a silent fall back to the math kernel |
| recompiles / graph breaks | dynamo counters + `torch._dynamo` log capture | a shape escaping the bucket set |

`torch.cuda.synchronize()` appears **only** in `xjepa/perf/` (contract hard rule 2).
On a machine with no GPU the harness runs, times the CPU step, and reports every
GPU-only field as unavailable with the reason — it never fabricates a number.

---

## 5. Reviewer checklist

Run these in order. Any "no" is a defect, not a preference.

**Structural**

- [ ] `grep -rn "DataLoader" xjepa/` returns nothing in the training path.
- [ ] `grep -rn "\.item()\|\.cpu()\|\.numpy()\|float(" xjepa/train/` returns nothing inside the step function.
- [ ] `grep -rn "cuda.synchronize" xjepa/` returns hits only under `xjepa/perf/`.
- [ ] `torch.compile` is called without `dynamic=True`, and bucket shapes are fixed.
- [ ] Attention goes through `F.scaled_dot_product_attention` only.
- [ ] AdamW is `fused=True` on CUDA (see the contract problem in §6).

**Measured**

- [ ] `python -m xjepa.perf.bench --steps 50 --warmup 10` completes and prints an MFU.
- [ ] **`h2d_bytes_per_step` is 0** (or a handful of bytes for a one-off constant).
      Anything in the megabytes means a batch is being built on the host.
- [ ] `implicit_syncs` is 0 over the measured window.
- [ ] `sdpa_backend` is `mem_efficient` (expected) or `flash` (better). `math` is a bug.
- [ ] fwd + bwd + opt accounts for >90% of the step; a large "other" means launch gaps
      or a stall between steps.
- [ ] `python -m xjepa.perf.profile_report --compile` reports **no recompiles** after
      warmup, and no `Memcpy HtoD` rows.
- [ ] Peak allocated memory leaves headroom for the 2.56 GB target bank plus
      activations — i.e. well under 24 GB.

**Reported**

- [ ] The MFU table states which N (7.41M / 7.51M / 8.10M) it used.
- [ ] The measured throughput was fed back into the £ budget before the grid was
      launched (RESEARCH_PLAN.md §6.1), and the pilot numbers are in the repo.
- [ ] Wall-clock per condition is reported alongside the fixed-token matching
      (RESEARCH_PLAN.md §3.3).

---

## 6. Contract problems found

1. **Hard rule 4 is impossible as written.** `docs/CONTRACTS.md` requires
   `foreach=True, fused=True` on AdamW. PyTorch raises
   `RuntimeError: `fused` and `foreach` cannot be `True` together.` — they select
   between two mutually exclusive implementations. The intent (use the fast path,
   not the per-parameter loop) is satisfied by `fused=True` on CUDA, which subsumes
   `foreach`. `xjepa/perf/` uses `fused=True` on CUDA and `foreach=True` on CPU.
   **Rule 4 should be amended to "fused=True on CUDA, foreach=True otherwise".**
2. **Rule 5 (SDPA-only) and the padded-bucket design fight each other.** Padding
   inside a bucket forces an attention mask, which forces the mem-efficient kernel
   and costs a modelled ~1.4× end-to-end (§3.1). The contract should say which of
   the two it prefers, and the choice should be recorded in the methods.
3. **`xjepa/eval/collapse.py` in the contract lists only the three free functions.**
   The trainer also needs the fixed held-out probe batch to live somewhere; that is
   `CollapseMonitor` in the same module, added without changing the three published
   signatures.

---

## 7. Measured numbers

Fill this table from the pilot before launching the grid; leave the modelled column
visible so the gap is auditable.

| Metric | Modelled | Measured (pilot) | Measured (grid) |
|---|---|---|---|
| steps/sec @ 128×512 | 10–11 | | |
| tokens/sec | 650k–720k | | |
| MFU (N = 7.41M) | 18–25% | | |
| H2D bytes/step | 0 | | |
| implicit syncs/step | 0 | | |
| fwd / bwd / opt (%) | ~33 / ~62 / ~1 | | |
| SDPA backend | mem_efficient | | |
| min/run (1B tokens) | 23–32 | | |
| £ for the 23-run grid | £1.7–2.4 | | |
