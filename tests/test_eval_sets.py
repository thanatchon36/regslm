import json
from pathlib import Path
from typing import Any

import pytest
from test_perturb_set import _ex, _unit

from regslm.corpus.base import Unit
from regslm.data.eval_sets import build_sets, write_sets
from regslm.eval.metrics import MemRow, memorisation_decomposition, memorisation_outcomes
from regslm.schema.obligation import Extraction
from regslm.train.evaluate import decomposition_ci, evaluate, paired_of1
from regslm.train.predict import REUSE_FROM, Item, assemble_set, order_sets

NUM = "Firms must hold 8 % capital within 30 days."


def _mk(i: int, text: str) -> tuple[Unit, Extraction]:
    u = _unit(text).model_copy(update={"unit_id": f"s/d/{i}"})
    ex = _ex(text, text, 8 if "8 %" in text else None, deadline="30 days" if "30" in text else "")
    return u, ex.model_copy(update={"unit_id": u.unit_id})


def _fixture() -> tuple[dict[str, Unit], dict[str, Extraction], list[dict[str, Any]]]:
    spec = [
        ("test", "en", True, NUM), ("test", "th", False, "Firms must report."),
        ("test", "en", False, "Firms must act."), ("dev", "en", False, NUM),
        ("silver_val", "en", False, NUM), ("heldout", "en", False, "Firms must act."),
        ("train", "en", False, NUM),
    ]  # fmt: skip
    units, labels, assign = {}, {}, []
    for i, (sp, lang, pc, text) in enumerate(spec):
        u, ex = _mk(i, text)
        units[u.unit_id], labels[u.unit_id] = u, ex
        assign.append({"unit_id": u.unit_id, "split": sp, "language": lang, "postcutoff": pc})
    return units, labels, assign


def test_build_and_write_sets(tmp_path: Path) -> None:
    units, labels, assign = _fixture()
    sets = build_sets(units, labels, assign, sample_n=1)
    n = {k: len(v) for k, v in sets.items()}
    assert n["test"] == n["test_perturbed"] == 3
    assert n["num440"] == n["num440_perturbed"] == 1
    assert n["mr_pool_s0"] == n["mr_pool_s1"] == 3  # test + dev + silver_val numeric; not train
    assert n["heldout"] == 1 and n["silver_val"] == 1 and n["silver_val_1k"] == 1
    assert n["postcutoff"] == 1 and n["thai_test"] == 1
    tp = {r["unit"]["unit_id"]: r for r in sets["test_perturbed"]}
    te = {r["unit"]["unit_id"]: r for r in sets["test"]}
    assert tp["s/d/1"]["unit"] == te["s/d/1"]["unit"]  # unchanged unit identical
    assert tp["s/d/0"]["unit"]["text"] != te["s/d/0"]["unit"]["text"]
    kinds = {t[1] for r in sets["mr_pool_s0"] for t in r["memorisation_targets"]}
    assert kinds == {"threshold.value", "deadline.value"}
    s0 = {r["unit"]["text"] for r in sets["mr_pool_s0"]}
    s1 = {r["unit"]["text"] for r in sets["mr_pool_s1"]}
    assert s0 != s1
    lp = tmp_path / "labels.jsonl"
    lp.write_text("x\n")
    man = write_sets(sets, tmp_path / "out", lp)
    assert len(man["labels_sha256"]) == 64 and man["sets"]["mr_pool_s0"]["n"] == 3
    assert (tmp_path / "out" / "MANIFEST.json").exists()
    assert json.loads((tmp_path / "out" / "num440.jsonl").read_text().splitlines()[0])["reference"]


def _mem(pred_thr: float | None, pred_dl: str) -> MemRow:
    from regslm.schema.obligation import Threshold

    ref = _ex(NUM, NUM, 12, deadline="45 days")
    pred = _ex(NUM, NUM, None, deadline=pred_dl)
    if pred_thr is not None:
        pred.obligations[0] = pred.obligations[0].model_copy(
            update={"threshold": Threshold(value=pred_thr, unit="%", comparator=">=")}
        )
    return MemRow(
        ref, pred, [[0, "threshold.value", 8.0, 12.0], [0, "deadline.value", 30.0, 45.0, "day"]]
    )


def test_decomposition() -> None:
    assert memorisation_outcomes(_mem(8, "30 days")) == ["original", "original"]
    assert memorisation_outcomes(_mem(12, "45 days")) == ["perturbed", "perturbed"]
    assert memorisation_outcomes(_mem(9, "10 days")) == ["other", "other"]
    assert memorisation_outcomes(_mem(None, "")) == ["missing", "missing"]
    assert memorisation_outcomes(_mem(12, "soon")) == ["perturbed", "missing"]
    assert memorisation_outcomes(_mem(12, "10 days")) == ["perturbed", "other"]
    assert (
        memorisation_outcomes(MemRow(_mem(1, "").reference, None, _mem(1, "").targets))
        == ["missing"] * 2
    )
    d = memorisation_decomposition([_mem(8, "30 days"), _mem(12, "45 days"), _mem(9, "")])
    assert d["original"] == d["perturbed"] == 2 / 6 and d["other"] == 1 / 6
    assert sum(d[c] for c in ("original", "perturbed", "other", "missing")) == pytest.approx(1)


def test_decomposition_ci_groups_by_unit() -> None:
    rows = [("a", _mem(8, "30 days")), ("a", _mem(12, "45 days")), ("b", _mem(8, "30 days"))]
    r = decomposition_ci(rows, n_boot=200)
    assert r["n_units"] == 2 and r["n_targets"] == 6
    assert r["original"]["point"] == pytest.approx(4 / 6)
    assert r["original"]["lo"] <= r["original"]["point"] <= r["original"]["hi"]


def test_assemble_set_dedupes() -> None:
    units, labels, _ = _fixture()
    mk = lambda i, u: Item(u, labels[f"s/d/{i}"].model_dump(mode="json"))  # noqa: E731
    base = {f"s/d/{i}": mk(i, units[f"s/d/{i}"]) for i in range(3)}
    base_rows = {k: {"completion": "BASE", "valid": False} for k in base}
    pu = units["s/d/0"].model_copy(update={"text": "Firms must hold 9 % capital."})
    items = [mk(0, pu), mk(1, units["s/d/1"]), mk(2, units["s/d/2"])]
    calls: list[int] = []

    def gen(its: list[Item]) -> list[dict[str, Any]]:
        calls.append(len(its))
        return [{"completion": "NEW", "valid": False} for _ in its]

    rows, n_run = assemble_set("test_perturbed", items, base, base_rows, gen)
    assert n_run == 1 and calls == [1]
    assert [r["completion"] for r in rows] == ["NEW", "BASE", "BASE"]
    assert all(r["set"] == "test_perturbed" for r in rows)
    rows, n_run = assemble_set("test_perturbed", items, {}, {}, gen)  # no base: run all
    assert n_run == 3


def test_order_sets() -> None:
    o = order_sets(["num440_perturbed", "thai_test", "test_perturbed", "heldout", "test"])
    for n in o:
        src = REUSE_FROM.get(n)
        if src in o:
            assert o.index(src) < o.index(n)
    assert sorted(o) == sorted(set(o)) and len(o) == 5


def _pred_row(i: int, set_: str, ex: Extraction, tg: list[Any], pred: Extraction) -> dict[str, Any]:
    return {
        "set": set_, "unit_id": f"u{i}", "valid": True, "extraction": pred.model_dump(mode="json"),
        "n_unresolved": 0, "reference": ex.model_dump(mode="json"), "latency_s": 0.1,
        "n_gen_tokens": 5, "memorisation_targets": tg,
    }  # fmt: skip


def test_evaluate_contamination_keys(tmp_path: Path) -> None:
    ref = _ex(NUM, NUM, 8, deadline="30 days")
    pert = _ex(NUM, NUM, 12, deadline="45 days")
    tg = [[0, "threshold.value", 8.0, 12.0], [0, "deadline.value", 30.0, 45.0, "day"]]
    empty = ref.model_copy(update={"obligations": []})
    d = tmp_path / "r" / "predictions"
    d.mkdir(parents=True)
    files = {
        "num440": [_pred_row(i, "num440", ref, [], ref) for i in range(6)],
        "num440_perturbed": [
            _pred_row(i, "num440_perturbed", pert, tg, ref if i % 2 else empty) for i in range(6)
        ],
        "mr_pool_s0": [_pred_row(i, "mr_pool_s0", pert, tg, ref) for i in range(4)],
        "mr_pool_s1": [_pred_row(i, "mr_pool_s1", pert, tg, pert) for i in range(4)],
    }
    for k, rows in files.items():
        (d / f"{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = evaluate("r", n_boot=100, runs=tmp_path)
    p = res["num440_paired"]
    assert p["n_units"] == 6 and p["of1_a"] == 1.0
    assert p["diff"] == pytest.approx(1 - 2 / 3) and p["lo"] <= p["diff"] <= p["hi"]
    dec = res["mr_pool_decomposition"]
    assert dec["n_units"] == 4 and dec["original"]["point"] == pytest.approx(0.5)
    assert dec["perturbed"]["point"] == pytest.approx(0.5)
    assert "memorisation_decomposition" in res["sets"]["mr_pool_s0"]
    assert paired_of1(files["num440"], files["num440"], 50)["diff"] == 0
