# Methods

<!-- Draft, 22 Oct 2026. Every number here is checked against the code by
     `python scripts/check_methods.py paper/methods.md`. Placeholders in
     [square brackets] are filled from the finished runs. -->

## Overview

We ask whether latent prediction learns useful protein sequence representations
without masked language modelling when the prediction targets come from a frozen,
pretrained structure encoder rather than from a jointly learned one. We pretrain
small sequence encoders under four objectives that differ only in their loss, with
identical data, architecture, optimiser and step budget, and compare them with
collapse diagnostics and frozen-feature probes.

## Pretraining corpus

Sequences and predicted structures come from the AlphaFold Database Swiss-Prot
subset. We fetched 79,446 unique accessions with mean pLDDT ≥ 70, excluding
disordered chains, whose structure targets would be noisy and near-degenerate.

**Redundancy reduction.** MMseqs2 clustering at 50% sequence identity and 80%
coverage left 33,855 cluster representatives.

**Pretrain/test separation (L0).** We removed any representative with ≥ 30%
identity over ≥ 50% coverage to any evaluation test sequence (SCOPe, secondary
structure, contact, fluorescence, stability). This removed 11,016
representatives. The final corpus is **22,839 chains, 8.90M residues**. Short
chains (< 128 residues) are 9.2% of the corpus; the filter hits them hardest,
consistent with SCOPe domains being short. We do not refill the short band, since
that would mean selecting pretraining data by its overlap with the test set.

**Length policy.** Chains shorter than 40 residues are excluded. Chains longer
than the 512-residue context are randomly cropped to a 512 window, redrawn each
pass, rather than discarded, following the convention of ESM-2.

## Structure targets

Per-residue targets are the encoder outputs of ESM-IF1
(`esm_if1_gvp4_t16_142M_UR50`; Hsu et al., 2022), computed from backbone
coordinates (N, Cα, C) of the full predicted structure. ESM-IF1's encoder reads
no amino-acid identities, so a target carries the local geometry of a masked
residue but not its label. Targets are computed once from the complete structure,
so a cropped window still receives targets that encode its whole-fold context.

The raw 512-dimensional embeddings are reduced by PCA to **256 dimensions**,
retaining **0.879** of the variance, and each component is standardised. The
effective rank (RankMe; Garrido et al., 2023) of the unwhitened bank is 229.8 at
256-d and 418.9 for the raw 512-d embeddings. A planned 128-d reduction retained
only 0.720 of the variance and was rejected before any training.

## Model

All conditions share an encoder in the ESM-2 t6 configuration with random
initialisation: 6 layers, model width 320, 8 attention heads, feed-forward width
1280, rotary position embeddings, a 33-token vocabulary and a 512-token context.
This is **7.41M parameters** (7,408,960). We use 8 heads rather than ESM-2's 20;
the parameter count is unchanged, and the pilot measured 1.24× higher throughput.

Conditions with a predictor (C2, C3) use a narrow transformer: 2 layers, width
160, 4 heads, feed-forward width 640, output projected to the 256-d target space.

## Objectives

Every masked condition selects **20%** of positions (Ofer et al., 2026) and applies
the same 80/10/10 corruption to them: 80% replaced by `<mask>`, 10% by a random
residue, 10% left unchanged. Applying the same corruption to the MLM and latent
conditions makes their inputs identically distributed, so the comparison isolates
the objective.

| id | objective | target | predictor | loss on |
|---|---|---|---|---|
| C1 | masked amino-acid cross-entropy | residue identity | no | masked positions |
| C2 | latent prediction, EMA target | EMA copy of the encoder and a linear head, on the uncorrupted sequence | yes | masked positions |
| C3 | latent prediction, frozen target | ESM-IF1 bank (PCA-256) | yes | masked positions |
| C5b | masked distillation | ESM-IF1 bank (PCA-256) | no, linear head | masked positions |

Latent objectives use cosine distance plus smooth-L1 (β = 1) on L2-normalised
vectors, weighted equally. C2's EMA momentum rises from 0.996 to 1.0 over training.
C2 has no anti-collapse term (no centring, whitening or variance penalty):
observing its collapse is the replication of Ofer et al. C5b is C3 without the
predictor, so the C3–C5b comparison isolates the predictor's contribution.

## Training

The configurations of all conditions are byte-identical outside the objective
block. Each run trains for **7,630 steps** of 65,536 token slots (128 sequences of
up to 512 tokens), about **486M real tokens** once padding is excluded
([range across runs]). The budget is fixed in steps and tokens, not wall-clock;
wall-clock per condition is reported as a secondary column.

Optimisation: AdamW, learning rate 4e-4, 500 warmup steps then cosine decay to
zero, weight decay 0.01, β = (0.9, 0.98), gradient clipping at 1.0, bf16 autocast
with fp32 master weights, `torch.compile`.

Sequences are batched into length buckets (128, 256, 384, 512) under a hybrid
policy: a sequence pads up to its bucket when occupancy is at least 0.85 and is
otherwise randomly cropped down to the bucket below. This gives 95.7% padding
efficiency and crops about 14% of residues per pass; crop windows are redrawn each
pass. The policy is identical across conditions.

The corpus and target bank are held in GPU memory for the whole run. Runs used
one RTX 4090 (24 GB). C1, C2 and C3 were trained with seeds 0, 1 and 2; C5b with
seeds 0 and 1.

## Collapse diagnostics

Every 500 steps, and on the final encoder output, we compute four statistics on a
fixed held-out probe batch:
- RankMe, the soft rank from the entropy of the singular values;
- the mean per-dimension standard deviation;
- the fraction of dimensions with standard deviation below 0.01;
- off-diagonal covariance mass.

These are measured on the **encoder** output, which is what the probes consume,
not on the predictor output. Reference lines are the target bank's RankMe and that
of a randomly initialised encoder.

## Evaluation

All probes use frozen encoder features. A single fixed hyperparameter grid per task
is selected on the validation split and applied identically to every condition;
nothing is tuned per condition.

**Secondary structure (SS3, SS8).** NetSurfP-2.0 data via TAPE: train on the
HHblits set (10,792 proteins), select on TS115 (115), test on CB513 (511 proteins;
the commonly quoted 513 reflects a curation difference). Per-residue probes over a
grid of three linear and three one-hidden-layer (256-unit) heads, 20 epochs each.
The primary metric is accuracy of the selected head.

**Fold retrieval.** SCOPe ASTRAL 2.08 at 40% identity (15,177 domains, 1,257
folds, 2,065 superfamilies). Zero-shot: mean-pooled embeddings are ranked by cosine
similarity, with queries and gallery split so that no superfamily appears on both
sides. A retrieval is correct when it shares the query's fold. The metric is top-1
accuracy, with MAP also reported.

**Baselines.** Every table includes:
- one-hot residue features;
- a randomly initialised encoder of the same architecture;
- the public ESM-2 t6 8M model;
- the raw ESM-IF1 embeddings of the evaluation proteins (the L1 ceiling).

For the ceiling, SCOPe structures are the real ASTRAL coordinates. CB513, TS115
and NetSurfP have no deposited structures, so their structures are predicted with
ESMFold. The SS training set for the ceiling probe is subsampled to 2,000 proteins.

## Hypotheses and analysis

Criteria were fixed before any result was seen (RESEARCH_PLAN.md §5.3, §9).

- **H1a, replication:** C2 collapses: its final encoder RankMe is below 25% of C3's, across all seeds.
- **H1b, ceiling:** C3's RankMe is reported against the target bank's 229.8; no pass/fail.
- **H2:** C3 ≥ C1 on at least 2 of 3 structure tasks (SS3, SS8, fold retrieval), with a seed-mean gap greater than one standard deviation.
- **H4, descriptive:** C3 vs C5b over the same tasks, reported without a verdict (see Deviations).

With three seeds, per-task p-values are not reported. We report the seed mean ± sd
per task, and the between-seed sd against the between-condition range as a result
in its own right.

## Deviations from the pre-registered plan

All of the following were decided on cost and time, before any grid result existed.

1. **Conditions:** four of seven trained (C1, C2, C3, C5b), 11 runs instead of
   23. C4 (MLM + λ·JEPA), C5, C5c, the λ sweep, the Foldseek 3Di target and span
   masking were not run. **H3 is not tested.** H4 needs both C5b and C5c over at
   least 5 tasks, so it is **reported descriptively** from C3 vs C5b only.
2. **Token budget:** about 0.5B tokens per run instead of 1.0B, identical across
   conditions. The plan sanctioned this reduction in advance as a fallback
   (§6.1).
3. **Tasks:** 3 of 7 evaluated. Contact prediction, fluorescence, stability and
   ProteinGym were dropped. Pre-registered predictions 4 and 5 are therefore
   untested. SS3 and SS8 are correlated, which makes H2's 2-of-3 count less
   independent than three unrelated tasks would be.
4. **Context:** 512 tokens rather than Ofer et al.'s 1024.
5. **Matching:** matched by tokens rather than wall-clock (Ofer et al. matched
   wall-clock). Matching wall-clock would penalise the predictor conditions for
   reasons unrelated to the objective.
6. **Seeds:** C5b has two seeds rather than three.

## Limitations

ESM-IF1 was trained on AlphaFold structures spanning UniRef50, so essentially every
evaluation protein is within its training distribution. L0 removes overlap between
pretraining and test data only; it cannot remove ESM-IF1's own exposure. C3 and C5b
therefore receive supervision from a 142M-parameter model trained on far more data
than C1 sees, an asymmetry that matching tokens does not fix. The claim this design
supports is about how best to spend a fixed sequence-model budget given access to a
pretrained structure encoder, not that latent prediction beats MLM at equal
information. The L1 ceiling probe quantifies how much of any gain ESM-IF1 already
contains.

<!-- TODO after results: wall-clock per condition, tokens_seen range, the GPU's
     measured throughput, any run that stopped on the wall-clock cap. -->
