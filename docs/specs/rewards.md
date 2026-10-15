# Spec: `regslm.rewards` — verifiable rewards for GRPO (paper §4.4)

Pure, deterministic, side-effect-free functions. Signature for each component:

```python
def <name>(completion: str, unit: Unit, reference: Extraction | None) -> float  # in [0, 1]
```

`completion` is the raw model output string; `unit` the input unit (text + id + source);
`reference` the silver label (None when the reward must be reference-free).
Each component parses `completion` once via a shared, memoised `parse(completion) ->
Extraction | None` (pydantic validation; strips a leading ```json fence; no repair).

| Component | Reference-free? | Definition |
|---|---|---|
| `schema_valid` | yes | 1.0 if `parse` succeeds, else 0.0. All other components return 0.0 when parse fails. |
| `citation_resolves` | yes | Mean over obligations of: `citation.unit_id == unit.unit_id` and `citation.source == unit.source` and `0 <= span.start < span.end <= len(unit.text)`. Empty extraction → 1.0. |
| `span_grounded` | yes | Mean over obligations of the fraction of non-empty text fields among {subject, action, object, condition, deadline, *exceptions} whose normalised form (guide §4) is a substring of the normalised unit text, or whose token-level overlap with the cited span's sentence is ≥ 0.8 (to tolerate lead-in-inherited subjects: subject may also match `unit.heading`). Empty → 1.0. |
| `numeric_exact` | no | Over obligations matched to the reference (guide §4 matching): fraction whose `threshold` equals the reference's (both None counts as equal). Additionally, any predicted threshold `value` that does not occur as a number in the unit text (after Thai-digit and thousands-separator normalisation) scores 0 for that obligation — this is what catches memorised values on perturbed text. No matched pairs → 1.0 if reference has no thresholds, else 0.0. |
| `deadline_exact` | no | As `numeric_exact` for `deadline` using the relaxed (number, time-unit) comparison. |
| `no_hallucinated_obligation` | no | `1 - FP / max(1, n_pred)` where FP = predicted obligations unmatched to the reference. If the reference is empty and the prediction is empty → 1.0. |

Composite (`config/rewards.yaml`):
```
R = w_schema·schema_valid · (w_cite·citation_resolves + w_ground·span_grounded
    + w_num·numeric_exact + w_dl·deadline_exact + w_hall·no_hallucinated_obligation) / Σw
    + recall_term
```
- The multiplicative `schema_valid` gate makes invalid JSON worth 0.
- **Anti reward-hacking** (RQ3): the precision-type components above are maximised by
  outputting `[]`. `recall_term = w_rec · matched / max(1, n_ref)` is therefore mandatory
  whenever a reference is given, and reported separately. Default weights: cite 1, ground 1,
  num 1, dl 0.5, hall 1, rec 1.5.
- Length guard: completions > 2× the reference's serialized length are clipped to 0 for
  `span_grounded` (prevents dumping the whole unit into fields).
- Logged per step: each component's batch mean, fraction of empty extractions, mean
  completion length, so degenerate solutions are visible (reported in RQ3 hacking checks).

Acceptance tests (`tests/test_rewards.py`): for each component, a perfect output, an invalid
JSON, an empty extraction, a hallucinated obligation, a wrong span, a paraphrased (ungrounded)
field, a memorised threshold on perturbed text (value absent from unit → 0); composite test
showing `[]` scores lower than a correct non-empty extraction on a unit whose reference has
obligations. ≥ 95 % line coverage on `rewards/`; mypy --strict clean.
