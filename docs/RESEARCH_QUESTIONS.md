# Pre-registered research questions

Status: **DRAFT** — becomes binding when tagged `prereg-v1` (before any training run beyond
`make quick`). Changes after the tag go to an "Amendments" section with date and reason; the
original text is never edited.

## 0. Common definitions

**Test sets.** `test` (in-distribution units, tier-P sources), `test-perturbed` (same units
after `perturb_numbers`, seed 0; labels updated through the change map), `heldout` (FCA SYSC,
fallback HKMA SPM — never in train/dev), `postcutoff` (units with `last_amended` after the
base model's documented training cutoff; N reported, may be small), `gold` (300 units, pending
owner adjudication; until then numbers on `gold` are reported as *pending*), `silver-val`
(2,000 teacher-labelled units held out from training, used while gold is pending).

**Primary metric — Obligation F1 (OF1).** Predictions and references matched by the
Hungarian assignment on citation-span IoU ≥ 0.5 with equal modality
(`regslm.rewards.normalize`); OF1 = 2PR/(P+R) over obligations, micro-averaged over units.
Units with empty reference and empty prediction contribute neither TP nor FP.

**Secondary metrics.** Field F1 per field (subject, action, object, condition, threshold,
deadline, exceptions) on matched pairs after guide §4 normalisation; Exact-obligation rate
(all fields equal); Numeric-exact (threshold and deadline); JSON validity; hallucination rate
(FP / predicted); latency, throughput, peak VRAM.

**Memorisation rate (MR).** On `test-perturbed`, among reference obligations with a perturbed
numeric value, the fraction where the model outputs the *original* (pre-perturbation) value.
Reported for every model × method × quantization cell.

**Headline rule.** Any claim about extraction ability uses OF1 on `test-perturbed`
("perturbation-corrected"). Unperturbed `test` is reported alongside, never alone.

**Statistics.** 3 seeds per cell (fewer only under the budget rule in §7). Unit-level
bootstrap (10,000 resamples) for 95 % CIs within a seed; across seeds, report mean ± 95 % CI
of the seed means (t-interval) and the pooled bootstrap. Paired comparisons use the paired
bootstrap over units on the same test set, averaged over seeds. Multiple comparisons across
sizes within one RQ: Holm correction, family-wise α = 0.05.

**Smallest effect of interest (SESOI).** 2.0 OF1 points (absolute). Differences with CI
inside ±2.0 are reported as "equivalent within SESOI", not as wins.

## RQ1 — Crossover and label efficiency

- **H1a.** Fine-tuned OF1 increases monotonically with N ∈ {500, 2k, 10k, 50k} for every
  size, with diminishing returns (gain 10k→50k < gain 2k→10k).
- **H1b.** A LoRA-SFT Qwen3-1.7B reaches the OF1 of Qwen3-8B zero-shot at N ≤ 2k, and of the
  14B teacher at N ≤ 10k (all on `test-perturbed`).
- **Refutation.** H1b is refuted for a comparison if, at all N, the CI of (student − baseline)
  includes or lies below zero. A crossover at N is *found* only if the CI excludes zero
  (student better) at N and at the next larger N.
- **Minimum N.** ≥ 1,500 test units per test set for OF1 CIs (expected half-width ≈ 2
  points); if `test` is smaller, report the achieved CI width.
- **Underpowered.** If the CI width exceeds 2 × SESOI, report the estimate as inconclusive
  and do not claim a crossover.
- Note: teacher-vs-student on silver labels is circular (students imitate the teacher). The
  teacher comparison is made on `gold` once adjudicated; until then it is reported as
  *pending* and on `silver-val` with that caveat.

## RQ2 — Rationale distillation vs label-only

- **H2a.** At equal *training token budget*, rationale-augmented SFT improves field F1 on
  `condition` and `exceptions` by ≥ SESOI over label-only, at 0.6B and 1.7B.
- **H2b.** The gain shrinks with size: gain(4B) < gain(0.6B).
- **Design.** Sizes 0.6B, 1.7B, 4B; N = 10k units; label-only and rationale runs matched on
  total target tokens (label-only gets more units to equalise; the matched N is recorded).
  Secondary contrast at equal *unit* count.
- **Refutation.** H2a refuted at a size if the paired CI of (rationale − label) on
  condition-F1 and exceptions-F1 both lie below +2.0. H2b refuted if gain(4B) ≥ gain(0.6B).
- **Minimum N.** ≥ 400 matched gold/silver-val obligations with a non-empty condition and
  ≥ 150 with exceptions; otherwise field-level results are descriptive only.

## RQ3 — GRPO with verifiable rewards

- **H3a.** GRPO from the best SFT checkpoint per size (0.6B, 1.7B, 4B) raises Numeric-exact
  by ≥ SESOI and lowers hallucination rate, without lowering OF1 by more than SESOI.
- **H3b.** GRPO lowers MR on `test-perturbed` relative to its SFT parent (the
  `numeric_exact` reward penalises values absent from the input).
- **Reward.** `config/rewards.yaml` as frozen at the tag; weights not tuned on any test set
  (tuned only on `silver-val`, with at most 3 configurations tried, all reported).
- **Hacking checks (always reported).** Fraction of empty extractions, mean completion
  length, per-component reward trajectories, and OF1 on `heldout` for SFT vs GRPO. A run is
  flagged *hacked* if empty-extraction rate rises > 10 points over SFT or recall drops > 5
  points while reward rises.
- **Refutation.** H3a refuted at a size if the Numeric-exact CI gain includes zero or OF1
  drops by > SESOI.

## RQ4 — Quantization and serving

- **H4a.** For each size, FP8 and AWQ-4 stay within SESOI of bf16 OF1; GGUF Q4_K_M loses
  more than SESOI at 0.6B but not at ≥ 4B.
- **H4b.** Where quantization hurts, Numeric-exact degrades before (more than) span-level OF1.
- **Measurements.** bf16 / FP8 / AWQ-4 / GPTQ-4 (vLLM), GGUF Q8_0 / Q4_K_M (llama.cpp CUDA
  and CPU-only), each: OF1, field F1, Numeric-exact, MR, tokens/s, p50/p95 latency per unit,
  peak VRAM / RAM. Speculative decoding 0.6B draft → 8B target: net throughput gain with
  identical outputs at temperature 0. Needle-in-rulebook at 4k/8k/16k (where fits).
- **Sanity rule.** A quantized model beating its bf16 parent by more than the CI width fails
  the build until explained in docs/ANOMALIES.md.
- **Refutation.** H4a refuted per (size, format) if the paired CI of (quant − bf16) lies
  entirely below −2.0 for FP8/AWQ, or does not for Q4_K_M at 0.6B.

## RQ5 — Thai vs English

- **H5a.** With the same schema and method, Thai OF1 is lower than English by ≥ SESOI at
  every size ≤ 4B for Qwen3.
- **H5b.** Thai-pretrained bases (Typhoon 2, SEA-LION v3) at ≤ 4B close at least half the
  Qwen3 Thai–English gap at equal size.
- **H5c.** English-only fine-tuning transfers zero-shot to Thai: OF1(Thai | en-trained) >
  OF1(Thai | base zero-shot) by ≥ SESOI.
- **Also reported.** Tokens per obligation and latency per document, Thai vs English.
- **Minimum N.** ≥ 500 Thai test units; if fewer, results are reported as exploratory.
- Caveat stated up front: Thai and English test units come from different instruments, so
  the gap mixes language and content effects; the parallel set (official translations) is
  used where available to separate them.

## Cross-cutting controls (mandatory, every reported model)

- MR on `test-perturbed`; held-out-rulebook gap (OF1 `test` − OF1 `heldout`).
- **Placebo:** SFT on shuffled labels (unit ↔ extraction permuted) at N = 10k, 0.6B and
  1.7B; expected OF1 near the zero-shot floor. If placebo ≥ real − SESOI, the training signal
  is suspect and the RQ1–RQ3 results are flagged.
- Sanity tests on every headline number (MISSION §0): exact 0/1/100 %, field F1 > 0.98, or
  quant > bf16 beyond CI → build fails until explained in docs/ANOMALIES.md.

## 7. Budget rule (pre-committed)

Total ≤ 100 GPU-hours, teacher ≤ 12. Cells are run in the priority order of MISSION §4. If
the projected total exceeds the cap: (1) drop seeds from 3 to 2 for 8B, then 4B; (2) drop
N = 50k for 8B; (3) drop GPTQ-4 where AWQ-4 exists. Every cut is listed in docs/PROGRESS.md
and the paper's Threats to Validity. No cut is chosen after looking at the results of the
cell it removes.
