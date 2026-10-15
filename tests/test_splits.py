import json
from pathlib import Path

import polars as pl

from regslm.data.splits import SplitConfig, assign, build_splits, fraction_deviation


def _units() -> pl.DataFrame:
    rows = []
    for doc in ("A", "B", "AMLD4"):
        for art in range(60):
            for par in range(5):
                rows.append(
                    {
                        "unit_id": f"s/{doc}/{art}/{par}/x",
                        "source": "s",
                        "doc_id": doc,
                        "language": "en",
                        "path": [f"Art {art}", f"{par}", "(a)"],
                        "last_amended": "2025-06-01" if art % 2 else None,
                    }
                )
    return pl.DataFrame(rows)


CFG = SplitConfig(heldout_docs=("s/AMLD4",), silver_val_n=40)


def test_rules_and_fractions() -> None:
    gold = {"s/A/0/0/x", "s/AMLD4/1/1/x"}
    a = assign(_units(), gold, CFG)
    assert a.height == 900
    sp = dict(zip(a["unit_id"], a["split"], strict=True))
    assert sp["s/A/0/0/x"] == "gold" and sp["s/AMLD4/1/1/x"] == "gold"
    assert set(a.filter(pl.col("doc_id") == "AMLD4")["split"]) == {"heldout", "gold"}
    assert all(abs(v) <= 0.01 for v in fraction_deviation(a, CFG).values())
    assert 30 <= a.filter(pl.col("split") == "silver_val").height <= 50
    # grouping: all units sharing group_id share a split (gold excepted)
    g = a.filter(pl.col("split") != "gold").group_by("group_id").agg(pl.col("split").n_unique())
    assert g["split"].max() == 1
    assert a.filter(pl.col("postcutoff")).height == 3 * 30 * 5


def test_deterministic_and_seeded() -> None:
    a1, a2 = assign(_units(), set(), CFG), assign(_units(), set(), CFG)
    assert a1.equals(a2)
    a3 = assign(_units(), set(), SplitConfig(seed=1, heldout_docs=("s/AMLD4",), silver_val_n=40))
    assert not a1.equals(a3)


def test_build_without_gold(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _units().write_parquet(corpus / "units.parquet")
    a = build_splits(CFG, corpus / "units.parquet", None, tmp_path / "none.jsonl", tmp_path / "o")
    assert a.filter(pl.col("split") == "gold").height == 0
    assert "VERSION" in (tmp_path / "o" / "SPLITS.md").read_text()
    (tmp_path / "g.jsonl").write_text(json.dumps({"unit_id": "s/A/0/0/x"}) + "\n")
    b = build_splits(CFG, corpus / "units.parquet", None, tmp_path / "g.jsonl", tmp_path / "o")
    assert b.filter(pl.col("split") == "gold").height == 1
