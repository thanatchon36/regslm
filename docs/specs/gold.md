# Spec: gold passes and adjudication file (paper §3.3)

- Inputs: `data/gold/sample.jsonl` (unit_id, stratum); units from `data/corpus/units.parquet`.
- Pass A: `regslm label --variant gold_a --model Qwen/Qwen3-14B-AWQ` → run `gold-pass-a`.
  Pass B: `regslm label --variant gold_b --model Qwen/Qwen3-8B-AWQ` → run `gold-pass-b`.
  (`scripts/export_units.py`-style export of gold units to `data/local/jobs/gold.jsonl` — add a
  `--unit-ids FILE` option to that script.) The engine config for the 8B: same defaults (it fits
  with more KV); nothing to change unless it fails.
- `regslm gold build` (new CLI group `gold` in cli.py, logic in `src/regslm/eval/gold.py`):
  reads both runs' `labels.jsonl`, computes `eval.agreement.inter_pass_agreement`, writes:
  - `data/gold/agreement.json` (all agreement numbers, N);
  - `data/gold/to_adjudicate.jsonl`: one row per unit **where the passes disagree** (any
    unmatched obligation, or any field/modality difference on matched pairs, or one pass
    invalid): {unit_id, stratum, language, heading, text, candidate_a, candidate_b,
    diff_summary (short human-readable list of differences), instruction: "Choose A, B, or
    write a corrected `gold` field (same JSON shape as candidate_a); see
    docs/ANNOTATION_GUIDE.md.", decision: null, gold: null}. Candidates as resolved
    `Extraction` JSON *plus* each obligation's quote text (`span` sliced from the unit text) so
    a human can read them.
  - `data/gold/agreed.jsonl`: units where both passes agree → provisional gold (still listed in
    the owner's review summary, not auto-final).
  - `data/gold/README.md`: how to adjudicate (edit `decision` to "A" | "B" | "edit" and fill
    `gold` when "edit"), and the command to finalise.
- `regslm gold finalize`: merges agreed + adjudicated rows → `data/gold/gold.jsonl`; refuses if
  any `decision` is null (prints count). Gold is never used for training or prompt tuning:
  add an assertion helper `assert_no_gold(unit_ids)` used by future training data builders.
Tests with tiny fake label files.
