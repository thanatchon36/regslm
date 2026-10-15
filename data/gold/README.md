# Gold set (paper 3.3)

`sample.jsonl` lists the 300 gold units (`unit_id`, `stratum`). Gold units are never used for
training or prompt tuning; training data builders must call `regslm.eval.gold.assert_no_gold`.

## 1. Export units and run the two passes

```
uv run --no-sync python scripts/export_units.py --unit-ids data/gold/sample.jsonl --splits gold --out data/local/jobs/gold.jsonl
make label RUN=gold-pass-a ARGS="--units data/local/jobs/gold.jsonl --variant gold_a"
make label RUN=gold-pass-b ARGS="--units data/local/jobs/gold.jsonl --variant gold_b --model Qwen/Qwen3-8B-AWQ"
```

## 2. Compare passes

```
uv run --no-sync regslm gold build
```

Writes `agreement.json` (all agreement numbers), `to_adjudicate.jsonl` (units where the
passes disagree) and `agreed.jsonl` (units where they agree: provisional gold, still listed
in the owner's review summary, not auto-final).

## 3. Adjudicate

Edit `to_adjudicate.jsonl`, one row per unit. Read `text`, `candidate_a`, `candidate_b`
(each obligation carries its `quote`) and `diff_summary`, following
`docs/ANNOTATION_GUIDE.md` (1. What counts as an obligation; 2. Fields; 3. Worked examples;
4. Normalisation for scoring). Then set `decision` to:

- `"A"` or `"B"`: accept that candidate (not possible if it is null, i.e. the pass was invalid);
- `"edit"`: fill `gold` with a corrected Extraction (same JSON shape as `candidate_a`).

## 4. Finalise

```
uv run --no-sync regslm gold finalize
```

Merges agreed and adjudicated rows into `gold.jsonl`; refuses (and prints the count) while
any `decision` is null.
