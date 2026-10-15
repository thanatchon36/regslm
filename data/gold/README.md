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

### 3b. Adjudicate in Excel (recommended) / ตัดสินผ่าน Excel

`gold build` sorts `to_adjudicate.jsonl` by tier (1 = existence/modality disagreement, 2 = only
field-boundary differences), then Thai first, then `unit_id`. Units whose passes agree under a
tolerant test (token-F1 tolerances) go to `agreed.jsonl`; strict numbers in `agreement.json` are
unchanged, tolerant counts sit beside them.

English:
1. `uv run --no-sync regslm gold export-csv` writes `data/gold/adjudicate.csv` (UTF-8 with BOM;
   also `adjudicate.xlsx` if openpyxl is installed).
2. Open the CSV in Excel. Read `text`, `A`, `B`, `differences`. Fill `decision` with `A`, `B` or
   `E` (edit). For `E`, write the corrected obligations in `note`, one per line, in the same
   format as A/B: `subject | modality | action | object | cond: … | thr: <= 5 % | dl: … | exc: x; y`
   (or just `none`). Modality is one of must, must_not, may, should.
3. Save as "CSV UTF-8 (Comma delimited)".
4. `uv run --no-sync regslm gold import-csv data/gold/adjudicate.csv` (reports parse errors and
   undecided rows; existing decisions are kept unless `--force`). Repeat 2-4 as often as needed.
5. `uv run --no-sync regslm gold finalize`.

ภาษาไทย:
1. รัน `uv run --no-sync regslm gold export-csv` จะได้ไฟล์ `data/gold/adjudicate.csv`
   (UTF-8 with BOM เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน)
2. เปิดใน Excel อ่านคอลัมน์ `text`, `A`, `B`, `differences` แล้วกรอกคอลัมน์ `decision` เป็น `A`, `B`
   หรือ `E` (แก้เอง) ถ้าเลือก `E` ให้เขียนข้อผูกพันที่ถูกต้องในคอลัมน์ `note` บรรทัดละ 1 ข้อ
   รูปแบบเดียวกับ A/B หรือเขียน `none` หากไม่มีข้อผูกพัน
3. บันทึกเป็น "CSV UTF-8 (Comma delimited)"
4. รัน `uv run --no-sync regslm gold import-csv data/gold/adjudicate.csv`
   (จะรายงานแถวที่ผิดรูปแบบและแถวที่ยังไม่ตัดสิน ไม่ทับค่าเดิมเว้นแต่ใช้ `--force`)
5. รัน `uv run --no-sync regslm gold finalize`

### Five rules that decide most cases / กฎ 5 ข้อที่ใช้บ่อย

- R1 named addressee: the subject must be a party named in the unit or its lead-in; never invent
  one. "shall" with no addressee (calculation/deeming) is none. / ผู้ปฏิบัติต้องระบุในข้อความหรือ
  lead-in ห้ามเติมเอง
- R3 deeming / scope / enabling: applicability, commencement, repeal, "deemed", delegation to
  adopt RTS are none (except a party told to treat/apply X: `must | treat/apply`).
- R7 lead-ins: a list point inherits subject and modality from a lead-in sentence naming a party
  ("A bank must:"); the point's content goes in object/condition; quote the point.
- Comparator rule: limit on a quantity = `must` with the compliant region (not exceed = <=);
  prohibited act above a limit = `must_not` with the prohibited region (>). Trigger numbers stay in
  `condition`.
- Thai: ให้ is `must` only as a clause-initial directive (ให้ + actor + action); not in ให้สินเชื่อ,
  เพื่อให้, ให้ใช้/ยกเลิก/ถือว่า. อาจ + party actor = `may`; อาจ + event = none; ควร = `should`.

## 4. Finalise

```
uv run --no-sync regslm gold finalize
```

Merges agreed and adjudicated rows into `gold.jsonl`; refuses (and prints the count) while
any `decision` is null.
