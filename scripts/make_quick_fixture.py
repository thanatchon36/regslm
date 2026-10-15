"""Build tests/fixtures/quick_labels.jsonl: 200 labelled rows with embedded units (pilot v3)."""

from __future__ import annotations

import json
from pathlib import Path

from regslm.data.splits import read_gold_ids

SRC = Path("results/runs/teacher-pilot-v3/labels.jsonl")
UNITS = [Path(f"data/local/units/{n}.jsonl") for n in ("eurlex", "ecfr", "thai")]
OUT = Path("tests/fixtures/quick_labels.jsonl")

units = {}
for p in UNITS:
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            units[r["unit_id"]] = r
gold = read_gold_ids(Path("data/gold/sample.jsonl"))
rows = []
for ln in SRC.read_text(encoding="utf-8").splitlines():
    r = json.loads(ln)
    if r["unit_id"] in units and r["unit_id"] not in gold:
        r["unit"] = units[r["unit_id"]]
        rows.append(r)
rows = rows[:200]
OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(len(rows), "rows ->", OUT)
