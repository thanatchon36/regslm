"""Export units of given splits to JSONL for `regslm label` (optionally trigger-only)."""

import argparse
import json

import polars as pl

from regslm.corpus.sample import add_trigger_column

ap = argparse.ArgumentParser()
ap.add_argument("--splits", help="comma-separated split names (ignored with --unit-ids)")
ap.add_argument("--unit-ids", help="JSONL with unit_id; selects exactly these units")
ap.add_argument("--out", required=True)
ap.add_argument("--trigger", choices=["any", "only", "none"], default="any")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

units = pl.read_parquet("data/corpus/units.parquet")
asg = pl.read_parquet("data/splits/assignments.parquet").select("unit_id", "split")
if a.unit_ids:
    with open(a.unit_ids, encoding="utf-8") as f:
        ids = [json.loads(ln)["unit_id"] for ln in f if ln.strip()]
    u = units.filter(pl.col("unit_id").is_in(ids))
elif a.splits:
    u = units.join(asg, on="unit_id").filter(pl.col("split").is_in(a.splits.split(",")))
else:
    ap.error("one of --splits or --unit-ids is required")
u = add_trigger_column(u)
if a.trigger == "only":
    u = u.filter(pl.col("trigger"))
elif a.trigger == "none":
    u = u.filter(~pl.col("trigger"))
u = u.sample(fraction=1.0, shuffle=True, seed=a.seed)
if a.limit:
    u = u.head(a.limit)
keys = [
    "unit_id",
    "source",
    "doc_id",
    "path",
    "heading",
    "text",
    "language",
    "version",
    "last_amended",
]
with open(a.out, "w", encoding="utf-8") as f:
    for r in u.select(keys).iter_rows(named=True):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(a.out, u.height)
