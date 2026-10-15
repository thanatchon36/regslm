# Spec: `regslm.eval` — metrics and statistics (paper §4.5; definitions in docs/RESEARCH_QUESTIONS.md §0)

Reuse `regslm.rewards.normalize` (normalisation, Hungarian matching on span IoU ≥ 0.5 with
equal modality, numeric helpers). Do not duplicate it.

## `eval/metrics.py`
Input: list of `(unit_id, reference: Extraction, prediction: Extraction | None, n_unresolved: int)`
(prediction None = invalid output; n_unresolved = generated obligations whose quote did not
resolve, D-020 — counted as false positives).
- `obligation_prf(...)` → P, R, OF1 micro over obligations (TP = matched pairs; FP = unmatched
  predictions + unresolved; FN = unmatched references; invalid output → all refs FN).
- `field_f1(...)` per field (subject, action, object, condition, threshold, deadline,
  exceptions): over matched pairs, a field counts TP if both non-empty and equal (normalised),
  FP if predicted non-empty and (gold empty or unequal), FN if gold non-empty and (pred empty or
  unequal); exceptions compared as sets of normalised strings. Unmatched obligations' non-empty
  fields also count FP/FN (so field F1 ≤ obligation-level coverage).
- `exact_obligation_rate`, `numeric_exact` (threshold + relaxed deadline on matched pairs with a
  numeric gold), `validity_rate`, `hallucination_rate` (FP / predicted), `empty_rate`.
- `memorisation_rate(rows)` where rows carry `memorisation_targets` (from
  `data/perturb_set.py`): fraction of targets where the prediction (any obligation matched to
  the target obligation) has the *original* value; also report `perturbed_correct_rate`.
- `per_unit_counts(...)` → polars DataFrame with per-unit TP/FP/FN (and per field) so that
  bootstrap resamples units cheaply.

## `eval/stats.py`
- `bootstrap_ci(per_unit_df, metric_fn, n=10_000, seed=0, alpha=0.05)` — resample units.
- `paired_bootstrap_diff(df_a, df_b, ...)` — same resampled unit indices for both systems;
  returns diff, CI, p (two-sided, proportion of resamples crossing 0 ×2).
- `seed_mean_ci(values)` — t-interval over seeds.
- `holm(pvalues)`.
- `crossover(n_values, diffs_with_ci)` — first N where CI excludes 0 (student better) at N and
  the next N (RQ1 rule); None otherwise.
- `sanity_flags(metrics: dict)` — returns list of anomalies per MISSION §0: any headline metric
  exactly 0 or 1, any field F1 > 0.98, quantized > bf16 parent by more than CI width (takes a
  parent mapping).

## `eval/agreement.py` (gold passes)
- `inter_pass_agreement(pass_a, pass_b)` → obligation-level span F1 between passes (treat A as
  reference; symmetric F1), field-level Cohen's κ per field computed over matched pairs
  (category = normalised value equality vs disagreement is not κ — instead: κ on *presence*
  (empty vs non-empty) per field, plus exact-agreement rate per field), modality κ over matched
  pairs, and unit-level agreement on "has any obligation".

Tests (`tests/test_eval.py`): hand-built references/predictions with known P/R/F1 and field
counts; invalid & unresolved handling; bootstrap CI contains the point estimate and is
deterministic by seed; paired diff of identical systems = 0 with CI [0,0]; holm on a textbook
example; crossover rule; sanity flags; κ on a textbook 2×2 table.
mypy --strict on src/regslm/eval; coverage ≥ 90 %.
