# Spec: `regslm.data.splits` — splits and perturbed test (paper §3.5)

Inputs: `data/corpus/units.parquet` (+ `data/local/corpus/units_local.parquet` if present),
`data/gold/sample.jsonl` (unit_ids; may be absent → treat as empty), config `config/splits.yaml`:
```yaml
seed: 0
heldout_docs: [eurlex/AMLD4]        # source/doc_id; D-023
test_frac: 0.10
dev_frac: 0.05
silver_val_n: 2000                  # drawn from the train side; never trained on
postcutoff_from: "2025-05-01"       # D-025
group_levels: 2                     # D-024: group by first N path levels within a doc
```
## Output `data/splits/` (committed; ids only, no text)
- `assignments.parquet`: unit_id, source, doc_id, language, group_id, split ∈ {train, dev,
  test, silver_val, heldout, gold}, postcutoff (bool).
- Rules: gold units → `gold` only. Held-out docs → `heldout` (minus gold). Others: groups
  (source/doc_id/path[:group_levels]) hashed with seed → test/dev/train by fraction, sizes
  checked within ±1 pp; `silver_val` = `silver_val_n` units sampled from train groups (whole
  groups, approximately N). `postcutoff` flag = `last_amended >= postcutoff_from` (any split);
  report counts per split.
- `SPLITS.md`: table split × source × language counts; postcutoff counts; VERSION hash.

## Perturbed test (`src/regslm/data/perturb_set.py`)
`perturb_unit_and_label(unit: Unit, ex: Extraction | None, seed) -> (Unit', Extraction' | None, changes)`:
- text' via `schema.perturb.perturb_numbers(unit.text, seed)`; heading perturbed too (separate call, its own change list).
- Spans: map each citation span through the edit list (offset shift by cumulative length
  delta of changes before it; a change inside a span extends/shrinks its end).
- Fields: in subject/action/object/condition/deadline/exceptions replace each changed original
  substring with its perturbed value (only when the original substring occurs in the field).
- threshold.value: if it equals the parsed numeric value of a change's original (use
  `rewards.normalize.extract_numbers`), set it to the perturbed value; record
  `memorisation_targets`: list of (obligation index, field, original value, perturbed value).
- Output file writer: `data/local/test_perturbed.jsonl` rows {unit (perturbed), label, changes,
  memorisation_targets}. (local because labels are derived from teacher output for now.)
Tests: span mapping incl. change before/inside/after span; threshold remap; EU comma; Thai
digits; units without numbers unchanged; determinism.

## CLI
`regslm split` → assignments + SPLITS.md. `regslm perturb-test --labels <labels.jsonl>` →
test_perturbed.jsonl (only units in split=test with a valid label).
Ruff clean; mypy --strict on both modules; tests in tests/test_splits.py, tests/test_perturb_set.py.
