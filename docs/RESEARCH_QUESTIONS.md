# Pre-registered research questions

Status: **DRAFT v2** (revised after an independent methodology review, D-037) — becomes binding
when tagged `prereg-v1` (before any training run beyond `make quick`). Changes after the tag go to
"Amendments" with date and reason; the original text is never edited. Grid: `config/grid.yaml` as
of the tag. Annotation rules: docs/ANNOTATION_GUIDE.md v2.0; teacher prompt v4.

## 0. Common definitions

### Test sets
- `test` — 4,407 units (en + th), in-distribution, tier-P sources. Labels are **silver** (14B
  teacher), so scores on it are "agreement with teacher labels", not accuracy.
- `test-perturbed` — `test` after `perturb_numbers` (seed 0), labels remapped (D-034). Only the
  440 units with numbers differ from `test`.
- `num440` — those 440 units, scored in original and perturbed form (paired).
- `mr_pool` — all 1,129 numeric units of test ∪ dev ∪ silver_val ∪ heldout, perturbed with seed 0
  for every model; a second independent draw (seed 1) is added for the cells where memorisation
  claims are made — zero-shot, the N = 10k SFT ladder and GRPO (D-041). Used only for
  memorisation metrics.
- `heldout` — EUR-Lex AMLD4, 641 units, never in train/dev (D-023); FCA SYSC added if obtained.
- `postcutoff` — units with `last_amended` ≥ 2025-05-01 (D-025); N reported.
- `gold` — 300 units, two-pass + owner adjudication. Until adjudicated, every gold-based verdict
  is "untested".
- `silver_val` — 2,000 teacher-labelled train-side units, used **only** for (1) choosing the
  "best SFT" checkpoint per size as the GRPO parent and the quantization checkpoint, (2) GRPO
  reward-weight checks (≤ 3 configurations, all reported). It is never used to select models
  reported for RQ1/RQ2, never reported as a test score of a model selected on it, and "best" is
  defined by `silver_val` OF1 alone. SFT checkpoints are final-step only (no early stopping).
- Thai test = `test` ∩ language th (581 units).

### Metrics
- **OF1 (primary).** Hungarian matching on citation-span IoU ≥ 0.5 with equal modality
  (`rewards.normalize`); micro P/R/F1 over obligations; unresolved quotes count as false
  positives (D-020); invalid output → all references false negatives.
- **Secondary:** field F1 (subject, action, object, condition, threshold, deadline, exceptions)
  plus unconditional recall of condition/exceptions; exact-obligation rate; Numeric-exact;
  validity; hallucination rate; latency, throughput, peak VRAM.
- **Memorisation decomposition** (on `mr_pool`, perturbation draws pooled where two exist): over reference
  obligations whose threshold **or** deadline value was perturbed, the fraction where the model's
  emitted value equals the **original** value (MR), equals the perturbed value (faithful), is a
  different value (other-wrong), or is not emitted (missing) — the four sum to 100 %. Reported for
  every model incl. the 14B teacher and zero-shot bases. "No memorisation" is claimed only if the
  upper 95 % CI bound of MR ≤ 5 %.

### Inference
- **Primary test:** paired unit-level bootstrap (10,000 resamples of units) on the per-unit
  score difference averaged over seeds. CIs are conditional on the observed seeds and the frozen
  corpus.
- **Seed consistency:** a contrast is "supported" only if the bootstrap criterion holds **and**
  the sign/threshold holds in every seed individually; otherwise "unsupported (seed-sensitive)".
  Per-seed values are always reported; no seed-level CI is claimed with 2–3 seeds.
- **SESOI:** 2.0 OF1 points on `test` / `test-perturbed`; 3.0 points for Thai and for field-level
  F1; 5.0 points on `num440` and on `gold`. If the achieved paired CI half-width exceeds the
  SESOI, the result is "inconclusive" and the hypothesis verdict is "not tested".
- **Equivalence** ("within SESOI") only via TOST: the 90 % paired CI lies inside ±SESOI.
- **Verdicts:** supported / refuted / inconclusive / not tested (cell not run, below minimum N,
  or gold pending). Refuted hypotheses are reported as refuted.
- **Holm families** (α = 0.05 each): RQ1 = per size, 2 baselines × 4 N; RQ2 = 2 fields × 3
  sizes (H2a); RQ3 = 3 sizes × 3 criteria; RQ4 = per size, the 5 non-bf16 formats; RQ5 = H5a over
  sizes. Everything else (incl. all secondary metrics) is exploratory and labelled as such.

### Baselines
Zero-shot baselines (Qwen3 0.6B–8B and the 14B teacher) receive the full teacher prompt v5 (rules
and examples) plus the generation JSON Schema in the prompt, without guided decoding — the
strongest instruction a user could give an un-tuned model, so crossover claims are not against a
strawman (D-043). Fine-tuned students receive the short student prompt.

### Headline rule (contamination)
Every OF1 is reported on `test` and `test-perturbed`. Claims about numeric extraction use
`num440` only: the paired difference original − perturbed (SESOI 5) plus the memorisation
decomposition on `mr_pool`. No claim of extraction ability rests on unperturbed numbers alone.

## RQ1 — Crossover and label efficiency

- **H1a.** For each size, seed-mean OF1 rises from N=500 → 2k → 10k → 20k, and the gain per
  log2(N) over 10k→20k is smaller than over 2k→10k. Supported if every step's paired CI lower
  bound > 0 and the paired bootstrap CI of the slope difference excludes 0 in that direction;
  refuted if any step decreases by more than SESOI with CI excluding 0.
- **H1b.** A LoRA Qwen3-1.7B is non-inferior to Qwen3-8B zero-shot at some N ≤ 2k (on `test` and
  `test-perturbed`), and to the 14B teacher at some N ≤ 10k **on `gold` only**.
- **Crossover rule.** At N, student is non-inferior to baseline B if the lower bound of the 95 %
  paired CI of (student − B) ≥ −SESOI. The crossover is the smallest N where this holds at N and
  at every larger evaluated N; if it first holds at 20k it is reported as "found at 20k,
  unconfirmed". H1b is refuted for B if the upper CI bound < −SESOI at every evaluated N.
- **Teacher circularity.** No "student ≥ teacher" claim is made on silver sets; silver
  agreement with the teacher is labelled "fidelity to teacher". The teacher's own silver-set
  scores are inflated by construction and are not compared.
- **Minimum N.** Achieved paired CI half-width is reported for every comparison.

## RQ2 — Rationale distillation vs label-only

- **H2a.** At equal training target tokens, rationale SFT improves condition F1 and exceptions
  F1 by ≥ 3.0 points (field SESOI) at 0.6B and 1.7B.
- **H2b.** The gain shrinks with size: gain(4B) < gain(0.6B), tested as one paired bootstrap on
  the difference of gains (unit resampling).
- **Design.** Rationale arm = 10k units with teacher rationales. Label arm = units added in the
  nested order until total target tokens match within ±5 % (cap 20k; if capped, reported as
  "under-matched"). Steps and tokens logged. Secondary contrast at equal units (the 10k ladder).
- Field metrics on `test` (silver; "teacher-agreement"); also unconditional recall of
  condition/exceptions to guard against matched-pair selection bias. Minimum: ≥ 400 reference
  obligations with a condition and ≥ 150 with exceptions in `test`; otherwise "not tested".

## RQ3 — GRPO with verifiable rewards

- **H3a.** GRPO from the best SFT checkpoint per size (0.6B, 1.7B, 4B) raises Numeric-exact on
  `num440` (perturbed) by ≥ 5 points, lowers hallucination rate, and does not lower OF1 on `test`
  by more than SESOI.
- **H3b.** GRPO lowers MR by ≥ 2 points (absolute) vs its SFT parent on `mr_pool`.
- **Reward** = `config/rewards.yaml` as frozen at the tag.
- **Hacking checks** (always reported): empty-extraction rate, completion length, per-component
  rewards, recall, `heldout` OF1. A run is flagged *hacked* if empty-rate rises > 10 pp or recall
  drops > 5 pp while reward rises; flagged runs are reported and excluded from the H3 verdict
  (verdict counts them as "not supported"), never tuned away.

## RQ4 — Quantization and serving

- **H4a.** Per size, FP8 and AWQ-4 are equivalent to bf16 (TOST within ±SESOI on `test`); GGUF
  Q4_K_M is not equivalent at 0.6B (paired CI of (Q4_K_M − bf16) excludes −SESOI) but is
  equivalent at ≥ 4B.
- **H4b.** Where a format loses more than SESOI, the loss on Numeric-exact (`num440`) is larger
  than the loss on OF1 (paired bootstrap on the difference of losses).
- **Measurements:** as in docs/specs/quant_bench.md (throughput, p50/p95 latency, VRAM/RAM;
  speculative decoding 0.6B → 8B with output identity check; needle-in-rulebook 4k/8k/16k
  where it fits; CPU-only for 0.6B/1.7B). Quantization uses the best-on-`silver_val` checkpoint
  per size.
- **Sanity rule:** a quantized model above its bf16 parent by more than the CI width fails the
  build until explained in docs/ANOMALIES.md.

## RQ5 — Thai vs English

- **H5a.** Same schema and method: Thai OF1 < English OF1 by ≥ 3.0 at every Qwen3 size ≤ 4B.
  Caveat: the Thai and English test units come from different instruments, so the gap mixes
  language and content; the FIBA parallel set (451 aligned pairs, exploratory) separates them.
- **H5b.** Thai-pretrained Typhoon 2 (1B, 3B) closes at least half the Qwen3 Thai–English gap at
  the nearest size (1B vs 0.6B/1.7B; 3B vs 1.7B/4B — "nearest size", not equal size): gap closure
  = (Thai OF1 Typhoon − Thai OF1 Qwen3) / (English OF1 Qwen3 − Thai OF1 Qwen3), CI by paired
  resampling of units; supported if the CI lower bound > 0.5.
- **H5c.** English-only fine-tuning (Qwen3-1.7B) transfers to Thai: OF1(Thai | en-trained) −
  OF1(Thai | zero-shot) ≥ 3.0 with CI excluding 0.
- Tokens per obligation and latency per document, Thai vs English (descriptive).
- **Minimum N** ≥ 500 Thai test units (581 available).

## 5. Cross-cutting controls (every reported model)

- **Memorisation decomposition** on `mr_pool` (§0).
- **Post-cutoff:** OF1 and MR on `postcutoff` vs the rest of `test`, with N; descriptive only if
  N < 200. Flag "contamination-suspect" if pre-cutoff − postcutoff OF1 > SESOI with CI excluding 0.
- **Held-out gap** = OF1(`test`) − OF1(`heldout`) with bootstrap CI; gap > 2 × SESOI flagged.
- **Placebo** (shuffled labels, seed 0, 0.6B and 1.7B, screening check): flag if placebo OF1 ≥
  real − SESOI (training signal suspect); flag "format learning" if placebo OF1 > zero-shot +
  SESOI. Zero-shot and placebo reported side by side with every real result.
- **Prompt-version sensitivity** (exploratory): teacher v3 vs v4 agreement on the 2,984 units
  labelled under both (D-036).
- **Sanity tests** (MISSION §0): a headline metric of exactly 0 / 1 / 100 %, field F1 > 0.98, or a
  quantized model above bf16 by more than the CI width fails the build until explained in
  docs/ANOMALIES.md.

## 6. Training protocol (fixed)

docs/specs/train.md defaults (D-029), epochs per D-035 (1 epoch for N ≥ 10k, 3 epochs for N ≤ 2k,
≥ 300 optimizer steps), nested label subsets, final checkpoint only. GRPO per docs/specs/grpo.md.

## 7. Budget rule (pre-committed)

- Total ≤ 100 GPU-h (teacher labelling, ≤ 12 GPU-h, counted separately). If the logged total
  reaches 100, stop; the cap is not soft.
- Re-projection uses throughput measured on the first N=10k seed-0 run of each size (all sizes)
  and logged GPU-hours only; `experiments/budget.py` never reads metric files. Projections and
  cut decisions are committed to docs/PROGRESS.md (with commit hash) **before** the affected
  block launches.
- Cuts already applied to the projection at pre-registration (D-035): 8B then 4B to 2 seeds; 8B
  N = 20k single seed; GRPO cap 1.5 h; Thai cells 2 seeds; SEA-LION 8B moved to optional block 8.
- Further cuts, only in this order: (1) drop block 8; (2) Thai cells to 1 seed; (3) GPTQ-4 where
  AWQ-4 exists; (4) 4B CPU/llama.cpp runs; (5) N = 20k for 4B to 1 seed; (6) 8B label-efficiency
  cells to 1 seed; (7) method_variants to 1 seed.
- A cut or truncated cell keeps its row in every results table as "not run" with verdict "not
  tested". A failed or OOM run is a result ("does not fit"); a crashed run may be re-run once with
  the same seed. Seeds are never replaced or dropped after their results are seen.

## Amendments
_None (pre-tag)._
