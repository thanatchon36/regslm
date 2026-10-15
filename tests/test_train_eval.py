from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from test_train import mk_label, mk_unit

from regslm.eval import metrics as M
from regslm.eval.stats import bootstrap_ci
from regslm.train.evaluate import bootstrap_all, evaluate, evaluate_rows
from regslm.train.predict import decode_row


def _row(i: int, valid: bool = True, wrong: bool = False) -> dict[str, Any]:
    u = mk_unit(i)
    ref = mk_label(u).extraction
    assert ref is not None
    pred = ref
    if wrong:
        pred = ref.model_copy(update={"obligations": []})
    return {
        "set": "t", "unit_id": u.unit_id, "valid": valid,
        "extraction": pred.model_dump(mode="json") if valid else None,
        "n_unresolved": 0, "reference": ref.model_dump(mode="json"),
        "latency_s": 0.5, "n_gen_tokens": 40, "memorisation_targets": [],
    }  # fmt: skip


def test_evaluate_rows_counts() -> None:
    rows = [_row(0), _row(1), _row(2, wrong=True), _row(3, valid=False)]
    r = evaluate_rows(rows, n_boot=50)
    m = r["metrics"]
    assert r["n_units"] == 4
    assert m["precision"] == 1.0 and m["recall"] == pytest.approx(0.5)
    assert m["validity_rate"] == 0.75
    assert r["ci95"]["of1"]["lo"] <= m["of1"] <= r["ci95"]["of1"]["hi"]


def test_bootstrap_all_matches_stats_bootstrap_ci() -> None:
    rows = [_row(i, wrong=i % 3 == 0) for i in range(12)]
    res = [
        M.UnitResult(r["unit_id"], mk_label(mk_unit(i)).extraction, None)
        for i, r in enumerate(rows)
    ]  # reference only; predictions are None -> all FN
    df = M.per_unit_counts(res)
    allci = bootstrap_all(df, n=100, seed=1)
    one = bootstrap_ci(df, M.metric("recall"), n=100, seed=1)
    assert allci["recall"] == pytest.approx(one)


def test_evaluate_writes_eval_json(tmp_path: Path) -> None:
    d = tmp_path / "r" / "predictions"
    d.mkdir(parents=True)
    (d / "quick.jsonl").write_text("".join(json.dumps(_row(i)) + "\n" for i in range(5)))
    res = evaluate("r", n_boot=20, runs=tmp_path)
    assert res["sets"]["quick"]["n_units"] == 5
    assert (tmp_path / "r" / "eval.json").exists()
    with pytest.raises(FileNotFoundError):
        evaluate("missing", runs=tmp_path)


def test_decode_row_valid_invalid_and_unresolved() -> None:
    u = mk_unit(1)
    ex = mk_label(u).extraction
    assert ex is not None
    from regslm.train.data import target_from_label, target_json

    ok = decode_row(target_json(target_from_label(ex, u.text)), u)
    assert ok["valid"] and ok["n_unresolved"] == 0 and len(ok["extraction"]["obligations"]) == 1
    assert decode_row("not json", u)["valid"] is False
    g = target_from_label(ex, u.text)
    g.obligations[0].citation.quote = "zzz qqq completely different"
    bad = decode_row(target_json(g), u)
    assert bad["valid"] and bad["n_unresolved"] == 1 and bad["extraction"]["obligations"] == []


def test_empty_frame_is_ok() -> None:
    assert isinstance(M.per_unit_counts([]), pl.DataFrame)
