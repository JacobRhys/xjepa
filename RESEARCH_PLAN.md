# Cross-Modal JEPA for Protein Sequences — Costed Research Plan (≤ £10 compute)

**Version:** 1.0 · 23 September 2026
**Constraint:** total compute spend < £10 (≈ $12.60 @ 1.26 USD/GBP)
**Deliverable:** 5-condition controlled pretraining study + probe suite, 3 seeds, reproducible on one consumer GPU.

---

## 0. Executive summary

The original protocol (35M–150M encoder, 5 conditions × 3 seeds, full TAPE + ProteinGym) costs on the order of
**hundreds of GPU-hours** and cannot be run for £10. This plan preserves *every hypothesis and every control*
by cutting three axes that do not change what the experiment tests:

| Axis | Original | This plan | Why the science survives |
|---|---|---|---|
| Encoder size | 35M (ESM-2 t12) | **8M (ESM-2 t6 config)** | All comparisons are *within-budget-matched*; absolute quality is not the claim. ESM-2 t6 is a published, real config (Lin et al., 2023), so the scale is defensible, not arbitrary. |
| Pretraining corpus | AFDB at scale | **50k AFDB Swiss-Prot chains ≈ 12.5M residues, ~1B training tokens** | Every condition sees identical data; collapse and probe-transfer are measurable at this scale. |
| Target dim | ESM-IF1 512-d | **512-d, PCA→128-d cached** | Cuts cache 4× (12.8 GB → 3.2 GB); variance retained is measured and reported. |
| Eval suite | Full TAPE + contact + ProteinGym | **SS3/SS8, fold retrieval, fluorescence, stability, contact-P@L/5, ProteinGym subset (~10 DMS)** | Frozen-feature probes; cost is dominated by one feature-extraction pass per model. |

**Costed total: ≈ $7.60 (≈ £6.05)**, leaving ~£4 headroom for reruns. A £0 fallback (Kaggle/Colab free tiers)
is specified in §7.

---

## 1. Scientific corrections to the protocol (read before building)

These are substantive and change how results must be interpreted. Each is cheap to address and strengthens the write-up.

### 1.1 H1 is close to tautological as written — reframe it

C3 regresses onto **fixed, externally-supplied vectors**. Representational collapse in the BYOL/I-JEPA sense is a
property of joint-embedding objectives with a *learnable* target; it is mathematically unavailable when the target is
frozen and non-degenerate. Finding "C3 does not collapse" is therefore predicted by construction, not evidence.

**Fix — make H1 falsifiable and informative:**

- H1a (*replication*): C2 (EMA target) collapses — RankMe(C2) → low. This is the thing that can actually fail.
- H1b (*upper bound*): RankMe(C3) ≤ RankMe(targets). A frozen target **caps** attainable rank at the effective rank
  of the target embeddings themselves. **Measure RankMe of the ESM-IF1 target bank first** and report it as the
  ceiling line on every rank plot. If target rank is e.g. 90/512, C3's representation is *rank-limited by supervision* —
  a real, publishable finding and a genuine risk to H2.
- H1c (*partial collapse*): report rank of the **encoder output**, not the predictor/projection output. The projection
  head can absorb rank; only the encoder representation is what probes consume.

### 1.2 The target leaks the masked residue's *geometry* but not its *identity* — state this explicitly

ESM-IF1's GVP-Transformer encoder consumes **backbone coordinates only** (N, Cα, C), not sequence (Hsu et al., 2022).
So the target at a masked position carries local backbone geometry of that residue but no amino-acid label. This is
a genuine difference from MLM and the core justification for the whole study — put it in the methods, and verify it
empirically (§2.3, leakage probe L1).

### 1.3 Context/target asymmetry differs from I-JEPA — make it a design choice, not an accident

In I-JEPA the target encoder sees *only* the target block. Here the frozen encoder sees the **whole structure**, so
targets at masked positions are contextualised by residues the sequence encoder cannot see. Two defensible variants:

- **T-full** (default): targets from the full structure. Simple, one cache, matches "distil structure" framing.
- **T-masked** (ablation, C3 only): recompute targets from the structure with masked-span coordinates deleted.
  Costs one extra target pass over a 10k subset. Run only if £ remains.

Report which you used. Reviewers will ask.

### 1.4 C5 as specified is not a tight control — add C5b

C5 (regress full sequence → structure embeddings, no mask, no predictor) differs from C3 in **three** ways at once
(masking, predictor, loss coverage). It cannot isolate the mechanism.

Add a ladder so exactly one factor changes per step:

| Cond | Mask? | Predictor? | Loss on | Isolates |
|---|---|---|---|---|
| C3 | ✓ 15% | ✓ | masked positions | (the treatment) |
| C5 | ✗ | ✗ | all positions | original control |
| **C5b** | ✓ 15% | ✗ (linear head only) | masked positions | value of the **predictor** |
| **C5c** | ✗ | ✓ | all positions | value of **masking** |

H4 is then tested as C3 vs C5b (predictor) and C3 vs C5c (masking), with C5 as the joint baseline.
C5b/C5c are the cheapest runs in the study (no extra data, same step count).

### 1.5 Loss scaling for λ in C4

MLM cross-entropy and L2-on-embeddings have different natural scales and different gradient magnitudes over training.
Do **not** hand-tune λ on 15 runs you cannot afford. Instead:

- Normalise targets to zero-mean/unit-variance per dimension over the cache (makes L2 ≈ O(1)).
- Use **cosine + smooth-L1** on L2-normalised vectors (as in I-JEPA/DINO practice) so the JEPA term is bounded in [0,2].
- Sweep λ ∈ {0.1, 1.0} on **seed 0 only**, pick by *pretraining-held-out JEPA loss + MLM ppl*, never by downstream score.
  Cost: 2 extra short runs. Report the sweep.

### 1.6 Leakage: be honest, it cannot be fully removed

ESM-IF1 was trained on ~12M AFDB structures spanning UniRef50 (Hsu et al., 2022). Essentially **every** eval protein is
inside its training distribution. A 30%-identity filter against "the encoder's training set" is not executable
(the set is effectively all of UniRef50).

**What is executable and should be done instead:**

- **L0 — pretrain/eval separation (mandatory):** MMseqs2 `easy-search`, filter any pretraining sequence with ≥30%
  identity + ≥50% coverage to any eval **test** sequence. Cheap (CPU, minutes) and it is the leakage that actually
  confounds probe results.
- **L1 — leakage probe (mandatory, cheap):** train a linear probe on the **raw ESM-IF1 target embeddings** for every
  downstream task. This gives the ceiling that C3/C4/C5 are distilling toward, and quantifies how much of any C3 gain
  is "ESM-IF1 already knows the answer."
- **L2 — fold-level splits:** for retrieval use SCOPe ASTRAL 2.08 40% with **superfamily-disjoint** query/gallery,
  not random splits.
- **L3 — disclosure:** state plainly that C3/C4/C5 receive supervision from a model trained on far more data than C1
  sees. **This is a compute-asymmetry that matched wall-clock does not fix.** The honest framing is:
  *"is latent structure prediction a better way to spend a fixed sequence-model budget, given access to a
  pretrained structure encoder?"* — not *"JEPA beats MLM at equal information."*

### 1.7 Statistical power with 3 seeds

3 seeds × ~8 tasks. Per-task t-tests are underpowered; do not report p-values per task as if they mean much.

- Primary: **mean ± sd per task**, plus per-task effect size (Cohen's d) with the caveat n=3.
- Aggregate: **paired sign test / Wilcoxon over tasks** on seed-averaged scores (n = #tasks), following Ofer et al.'s
  win/loss counting. State that task-level independence is an approximation.
- Report **seed variance as a first-class result**: if between-seed sd exceeds between-condition gaps, that *is* the
  finding, and it is the single most likely outcome at 8M scale. Pre-register this.

---

## 2. Objective O1 — pipeline (costed)

### 2.1 Data acquisition (CPU, £0, bandwidth only)

- Source: **AlphaFold DB Swiss-Prot v4** subset. Per-accession fetch:
  `https://alphafold.ebi.ac.uk/files/AF-{ACC}-F1-model_v4.pdb`
- Sample **50,000** accessions, filters: length 40–512, **mean pLDDT ≥ 70** (drop disordered chains — they give
  noisy, near-degenerate structure targets and will artificially depress target rank).
- Redundancy: MMseqs2 `easy-cluster` at 50% identity → keep cluster representatives. Prevents memorisation and
  makes the 1B-token budget cover more distinct structure.
- Apply **L0** filter against all eval test sets.
- Expected yield after filters: ~35–45k chains, ~10M residues.
- Time: ~6–10 h wall-clock of polite serial/8-way-parallel HTTP, run on a laptop. **£0.**

### 2.2 Target embedding cache (GPU, ~3–4 h)

- Model: `esm_if1_gvp4_t16_142M_UR50` — use **encoder only**, backbone coords in, 512-d per residue out.
  Licence: CC-BY-NC 4.0 — fine for academic work, **state it**.
- fp16 inference, batch by length bucket.
- Throughput estimate: ~250–400 chains/s-equivalent is optimistic; budget **3 h on RTX 4090** for 40k chains and
  measure on a 500-chain pilot before committing.
- **Post-processing (do this, it is what makes the cache affordable):**
  1. Fit PCA on a 2M-residue random sample → keep **128 components**; record explained variance (expect 85–95%).
  2. Standardise each component to unit variance.
  3. Store **fp16, 128-d** → 10M × 128 × 2 B = **2.6 GB**. Fits on any pod disk; push to a HuggingFace dataset (free) so
     reruns never recompute.
- **Report PCA explained variance and RankMe of the 128-d cache.** If PCA-128 loses meaningful rank, fall back to 256-d
  (5.1 GB, still fine).

### 2.3 Cheaper fallback target (if ESM-IF1 is a problem)

**Foldseek 3Di** tokens: a 20-letter structural alphabet derived from backbone geometry, computed on CPU in minutes for
40k chains, ~10 MB of storage. Gives a *discrete* structural target. Use as:
- a robustness check that results are not an ESM-IF1 artefact, and
- a £0 replacement if the ESM-IF1 cache overruns budget (C3 becomes cross-entropy on 3Di at masked positions).
Worth running regardless — it costs essentially nothing and doubles the evidence base for H2/H4.

---

## 3. Model and training configuration

### 3.1 Encoder — ESM-2 t6 8M config, random init

| Param | Value |
|---|---|
| Layers | 6 |
| d_model | 320 |
| Heads | 20 |
| FFN | 1280 |
| Pos. emb. | RoPE (as ESM-2) |
| Vocab | 33 (ESM-2 alphabet) |
| Params | ≈ 8M |
| Max len | 512 |

### 3.2 Predictor (C2, C3, C4, C5c)

2 layers, d=160, 4 heads, FFN 640 (≈ 0.6M params), + learned `[MASK]` position query embeddings.
Depth/width well below encoder, per I-JEPA. Linear 160→128 projection to target dim.

### 3.3 Training budget (identical across conditions — this is the control)

| Param | Value |
|---|---|
| Tokens/run | **1.0B** (fixed-token matching, *not* wall-clock — see note) |
| Batch | 128 seqs × 512 tok = 65,536 tok/step |
| Steps | ~15,300 |
| Optimiser | AdamW, lr 4e-4, warmup 500, cosine decay, wd 0.01, β=(0.9,0.98), clip 1.0 |
| Precision | bf16, `torch.compile`, flash attention |
| Mask rate | 15% (ESM-2 80/10/10 for MLM; for JEPA, masked positions replaced by `[MASK]`) |
| Bucket policy | `hybrid`: pad up when bucket occupancy ≥ 0.85, else random-crop down |

> **An "epoch" is not every residue once.** Token-budget batching fixes tokens per step, not
> padding: naive pad-up gives only 83.2% padding efficiency on a realistic length distribution
> (a 130-residue chain pads to 256). The `hybrid` policy reaches **95.7% efficiency at the cost
> of cropping 14.2% of residues per pass**. Crop windows are redrawn each pass so coverage is
> reached across passes, not within one. The fixed budget is **tokens**, so this does not change
> the information budget, and the policy is identical across conditions so it is controlled — but
> it must be stated in the methods, and runs report tokens seen, never epochs.

> **Fixed tokens, not fixed wall-clock.** Ofer et al. matched wall-clock; at this scale that would penalise conditions
> with a predictor for reasons unrelated to the objective. Fix tokens (the fair information budget), and **report
> wall-clock per condition as a secondary column** so both matchings are visible. Say this explicitly in methods;
> it is a deliberate deviation.

### 3.4 Throughput and per-run cost

**Revised after implementation — see `docs/PERF.md`.** The encoder is **7,408,960** parameters,
not 8M (ESM-2 t6's "8M" is a round-up), so FLOPs ≈ 6 · N · D = 6 × 7.41e6 × 1e9 = **4.45e16**.

The 25% MFU assumption was optimistic. Modelled realistic range on a 4090 is **18–25%**:

- **Attention dominates, not kernel launch.** It is 21% of step FLOPs but ~55% of step time.
  Two compounding causes: the key-padding mask disqualifies the flash backend so we land on
  cutlassF, and `head_dim = 320/20 = 16` leaves the tensor-core tile more than half empty.
- **GEMM shape.** K=320 is compute-bound (arithmetic intensity ≈ 254 vs the 4090's ~164 ridge),
  but the short reduction caps throughput near 40–50% of peak.
- Kernel launch overhead is ~2–3% of a ~100 ms step — real but not the problem.

→ **20 min per run at 25% MFU, 28 min at 18%.** Grid cost moves from $2.62 to ~$3.60 worst case.
Still inside budget, with less contingency than §6 assumes. Recompute from the pilot before
launching (§6.1).

**Open lever worth ~1.4× end-to-end.** Zero intra-bucket padding removes the attention mask
entirely, which makes the flash backend eligible. Achievable by cropping every sequence to
exactly its bucket length instead of the default `hybrid` pad/crop policy — costs more cropped
residues per epoch (22.6% vs 14.2%) in exchange for a materially faster step. Decide from the
pilot; whichever is chosen must be identical across all conditions.

---

## 4. Run grid

| # | Cond | Objective | Seeds | Runs |
|---|---|---|---|---|
| 1 | C1 | MLM only | 3 | 3 |
| 2 | C2 | JEPA, EMA target (τ 0.996→1.0) | 3 | 3 |
| 3 | C3 | JEPA, frozen ESM-IF1 target, masked-only loss | 3 | 3 |
| 4 | C4 | MLM + λ·JEPA | 3 | 3 |
| 5 | C5 | Distil, no mask, no predictor, all positions | 2 | 2 |
| 6 | C5b | Masked distil, no predictor | 2 | 2 |
| 7 | C5c | Unmasked distil, with predictor | 2 | 2 |
| 8 | λ sweep | C4 @ λ∈{0.1,1.0}, seed 0 | — | 2 |
| 9 | 3Di target | C3 variant, Foldseek target | 2 | 2 |
| 10 | span mask | C3 + span masking (mean span 8) | 1 | 1 |

**Total: 23 runs × 20 min ≈ 7.7 GPU-hours.**

Priority tiers if it overruns: **Tier 1** = rows 1–3 (C1/C2/C3, 3 seeds) — minimum viable study, answers H1a/H2.
**Tier 2** = rows 4, 6, 7 (H3, H4). **Tier 3** = rows 5, 8, 9, 10.

---

## 5. Evaluation (O2, O3)

### 5.1 Collapse diagnostics — logged every 500 steps

- **RankMe** (Garrido et al., 2023): soft rank from singular-value entropy of a fixed 8,192-embedding probe batch
  (held-out, same batch every time). Log for **encoder output** and separately for predictor output.
- **VICReg variance criterion**: mean per-dim std (Bardes et al., 2022), plus fraction of dims with std < 0.01.
- **Off-diagonal covariance mass** — catches dimensional collapse RankMe can smooth over.
- **Reference lines:** RankMe of the ESM-IF1 target bank, and of a randomly-initialised encoder (upper/lower anchors).

Cost: negligible (one forward pass on 8k residues per log point).

### 5.2 Downstream probes — frozen features, one extraction pass per model

Extract once per checkpoint, cache to disk, then all probes are CPU/seconds-scale sklearn or a 1-epoch torch head.

| Task | Source | Probe | Metric |
|---|---|---|---|
| Secondary structure (SS3/SS8) | TAPE / NetSurfP-2.0 CB513 + TS115 | linear, per-residue | accuracy |
| Contact prediction | TAPE ProteinNet (CASP12 test) | bilinear on residue pairs, **subsample 20k pairs/protein** | P@L/5 (medium+long) |
| Remote homology / fold | SCOPe ASTRAL 2.08 40%, superfamily-disjoint | **zero-shot** cosine retrieval on mean-pooled embeddings | top-1 acc, MAP |
| Fluorescence | TAPE | linear ridge on mean-pool | Spearman ρ |
| Stability | TAPE | linear ridge on mean-pool | Spearman ρ |
| Mutation effect | ProteinGym, **10 DMS subset** (stratified: viral/prokaryote/human, len < 400) | zero-shot: cosine(WT emb, mut emb) at mutated position | Spearman ρ |

**Probe protocol (identical across all conditions — this is what makes it a controlled comparison):**
- Encoder frozen. Single hyperparameter grid per task, selected on the *validation* split, applied identically to
  every condition. Never tune per condition.
- Report both linear and 1-hidden-layer (256 units) MLP probe; linear is primary.
- **Baselines on every table:** (a) one-hot/BLOSUM features, (b) randomly-initialised encoder, (c) raw ESM-IF1 target
  embeddings (the L1 ceiling), (d) public ESM-2 t6 8M (sanity anchor — shows whether 1B tokens is enough to learn anything).

> **Note on ProteinGym by embedding distance.** Cosine distance is not a fitness score and correlates weakly even for
> good models; expect |ρ| < 0.2. Pre-register this as **exploratory**, report it as such, and do not let H2 rest on it.
> C1 (MLM) *can* be scored by pseudo-likelihood — report that number too, clearly labelled as a different scoring
> function, not a like-for-like comparison.

### 5.3 Analysis → hypothesis map

| Hypothesis | Test | Pass criterion (pre-registered) |
|---|---|---|
| H1a | RankMe(C2) vs RankMe(C3) over training | C2 final rank < 25% of C3, across all 3 seeds |
| H1b | RankMe(C3) vs RankMe(targets) | reported as ceiling; no pass/fail |
| H2 | C3 vs C1 on SS3, contact, fold retrieval | C3 ≥ C1 on ≥2/3 structure tasks, seed-mean gap > 1 sd |
| H3 | C4 vs max(C1, C3) | C4 best on ≥5/8 tasks (sign test over tasks) |
| H4 | C3 vs C5b (predictor) and C3 vs C5c (masking) | C3 > both on ≥5/8 tasks ⇒ H4 supported; C3 ≈ C5 ⇒ H4 rejected, report as distillation result |

---

## 6. Costing

Prices: RunPod **Community Cloud RTX 4090 ≈ $0.34/h** (verify at booking — prices move; A5000 ≈ $0.26/h and
RTX 3090 ≈ $0.22/h are cheaper fallbacks, ~1.6× slower).

| Line | Hours | Rate | Cost |
|---|---|---|---|
| Pilot / throughput calibration / debugging | 2.0 | $0.34 | $0.68 |
| ESM-IF1 target cache (40k chains) | 3.5 | $0.34 | $1.19 |
| Pretraining grid (23 runs × 20 min @ 25% MFU) | 7.7 | $0.34 | $2.62 |
| — same grid at the pessimistic 18% MFU | 10.7 | $0.34 | $3.64 |
| Feature extraction for probes (23 ckpts × ~4 min) | 1.6 | $0.34 | $0.54 |
| Probe training (contact probe is the only GPU one) | 1.5 | $0.34 | $0.51 |
| Network volume 20 GB, 1 month | — | $0.07/GB/mo | $1.40 |
| **Subtotal** | **16.3** | | **$6.94** |
| Contingency (reruns, one failed condition) | 2.0 | $0.34 | $0.68 |
| **Total** | | | **$7.62 ≈ £6.05** |

CPU work (download, MMseqs2, Foldseek, sklearn probes, PCA) runs on the laptop: **£0**.
Storage of the released artefacts on HuggingFace Hub: **£0**.

**Headroom: ≈ £3.95.** Spend priority if it materialises: (1) 3rd seed for C5/C5b/C5c, (2) T-masked target ablation,
(3) 2B-token runs for Tier-1 conditions to check whether conclusions are budget-dependent.

### 6.1 Cost controls (enforce these or the budget fails)

- **Never leave a pod running.** Wrap every job so the container exits on completion; use a spot/community instance and
  checkpoint every 1,000 steps so preemption costs ≤ 1.5 min.
- Hard wall-clock killer per run (`timeout 35m`).
- Log to **Weights & Biases free tier** or plain CSV — no paid tooling.
- Do the 200-step pilot and **recompute the whole budget from measured throughput** before launching the grid.
  If measured MFU is below 15%, drop to 500M tokens/run and say so.

---

## 7. £0 fallback path

Everything above fits Kaggle free tier: **30 GPU-hours/week**, P100 16 GB or 2×T4. Caveats and adjustments:

- T4 has no bf16 → use fp16 + loss scaling; expect ~2.5× slower than 4090 ⇒ ~50 min/run. 23 runs ≈ 19 h — fits one week's
  quota with room for the target cache.
- 9 h/session limit ⇒ checkpoint/resume is mandatory (needed anyway).
- Kaggle datasets (max 100 GB) hold the 2.6 GB target cache fine.
- Colab free is *not* recommended: no quota guarantee, and disconnects will cost more in lost work than £6 saves.

Recommendation: **develop and debug on Kaggle free, run the final grid on a paid 4090** for clean, uninterrupted,
citable wall-clock numbers. That is what the £6 buys.

---

## 8. Timeline

| Week | Work | Output |
|---|---|---|
| 1 | AFDB download, MMseqs2 clustering, L0 leakage filter, eval datasets assembled | `data/` frozen, splits committed |
| 2 | ESM-IF1 cache + PCA; RankMe of target bank; Foldseek 3Di cache | `targets_128d.fp16`, ceiling numbers |
| 3 | Model/trainer/losses; 200-step pilot; **budget recomputation**; unit tests on masking & loss | measured throughput, final token budget |
| 4 | Tier-1 grid (C1/C2/C3 × 3 seeds); collapse diagnostics | H1a, H2 preliminary |
| 5 | Tier-2 + Tier-3 grid; λ sweep | H3, H4 |
| 6 | Feature extraction, all probes, baselines (incl. L1 ceiling) | full results tables |
| 7 | Analysis, sign tests, plots; leakage/limitations section | draft |
| 8 | Buffer / reruns / write-up | final |

---

## 9. Pre-registered predictions (commit before running)

Write these down now; they make a null result publishable.

1. C2 collapses (RankMe < 20% of C3). **High confidence.**
2. C3 does not collapse but is **rank-capped** near the target bank's effective rank. **High confidence.**
3. C3 > C1 on secondary structure and contact prediction. **Medium.** C3 is receiving structure supervision from a
   142M-param model; the surprise would be if it did not.
4. C3 < C1 on fluorescence/stability/ProteinGym. **Medium-high.** Structure targets carry little fitness signal.
5. C4 best overall. **Medium.**
6. **C3 ≈ C5 (H4 rejected).** **This is the modal outcome.** At 1B tokens and 8M params the model likely lacks capacity
   for the predictor to add anything beyond what direct regression gives. Plan the write-up so this is a *result*
   ("structure supervision, not latent prediction, drives the gain"), not a failure.
7. Between-seed sd comparable to between-condition gaps on the function tasks. **Medium-high.** Say so up front.

---

## 10. Reproducibility deliverables

- Single repo, `configs/{c1..c5c}.yaml` differing **only** in the objective block — diffable proof of the control.
- Fixed seeds, `torch.use_deterministic_algorithms(True)` where it does not cost >10% throughput.
- Released: accession list, cluster assignments, all splits, PCA basis, target cache (HF dataset), all checkpoints
  (23 × ~32 MB = 0.8 GB), per-step metric CSVs.
- One `make all` that reproduces every table from the cached features.

---

## 11. Key risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| 8M params too small to show any condition differences | Medium | Baselines (b) random encoder and (d) public ESM-2 8M bound the measurable range; if C1 ≈ random encoder, the budget is too small — report and raise tokens, not params |
| ESM-IF1 cache overruns time/disk | Medium | Pilot on 500 chains first; PCA-128; Foldseek 3Di fallback (§2.3) |
| Target rank is low ⇒ C3 rank-capped and weak everywhere | Medium | Measured up front in week 2, before any pretraining is spent — decision point, not a surprise |
| Leakage criticism | High | §1.6 L0–L3, stated openly as a limitation of the framing |
| Spot preemption | Medium | 1,000-step checkpoints; resume script |
| Budget overrun | Low | Post-pilot budget recomputation gate in week 3 |

