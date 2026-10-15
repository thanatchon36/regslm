import json
from pathlib import Path

import pytest

from regslm.eval import gold

TEXT = "The firm must report incidents within 72 hours. It may appeal."


def _ob(action: str = "report incidents", modality: str = "must", dl: str = "72 hours") -> dict:
    return {
        "subject": "The firm",
        "modality": modality,
        "action": action,
        "deadline": dl,
        "citation": {"source": "s", "unit_id": "u", "span": {"start": 0, "end": 48}},
    }


def _row(uid: str, obs: list[dict] | None) -> dict:
    if obs is None:
        return {"unit_id": uid, "valid": False, "extraction": None, "n_unresolved": 0}
    return {
        "unit_id": uid,
        "valid": True,
        "n_unresolved": 0,
        "extraction": {"unit_id": uid, "obligations": obs},
    }


def _w(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, dict, Path]:
    ids = ["agree", "empty", "modal", "invalid", "extra"]
    sample = tmp_path / "sample.jsonl"
    _w(sample, [{"unit_id": u, "stratum": "x"} for u in ids])
    a = [
        _row("agree", [_ob()]),
        _row("empty", []),
        _row("modal", [_ob()]),
        _row("invalid", [_ob()]),
        _row("extra", [_ob()]),
    ]
    b = [
        _row("agree", [_ob()]),
        _row("empty", []),
        _row("modal", [_ob(modality="should")]),
        _row("invalid", None),
        _row("extra", [_ob(), _ob("appeal", "may", "")]),
    ]
    _w(tmp_path / "a/labels.jsonl", a)
    _w(tmp_path / "b/labels.jsonl", b)
    units = {u: {"text": TEXT, "heading": "H", "language": "en"} for u in ids}
    return tmp_path / "a", tmp_path / "b", sample, units, tmp_path / "out"


def test_build_and_finalize(tmp_path: Path) -> None:
    a, b, sample, units, out = _setup(tmp_path)
    s = gold.build(a, b, sample, units, out)
    assert s["n_agreed"] == 2 and s["n_to_adjudicate"] == 3
    assert json.loads((out / "agreement.json").read_text())["n_both_passes"] == 5
    rows = [json.loads(ln) for ln in (out / "to_adjudicate.jsonl").read_text().splitlines()]
    by = {r["unit_id"]: r for r in rows}
    assert set(by) == {"modal", "invalid", "extra"}
    assert by["modal"]["candidate_a"]["obligations"][0]["quote"] == TEXT[:48]
    assert by["modal"]["diff_summary"]
    assert by["invalid"]["candidate_b"] is None
    assert any("only B" in d for d in by["extra"]["diff_summary"])
    assert all(r["decision"] is None and r["gold"] is None for r in rows)

    with pytest.raises(ValueError, match="3 of 3"):
        gold.finalize(out)

    for r in rows:
        r["decision"] = {"modal": "B", "invalid": "A", "extra": "edit"}[r["unit_id"]]
        if r["decision"] == "edit":
            r["gold"] = r["candidate_b"]
    _w(out / "to_adjudicate.jsonl", rows)
    assert gold.finalize(out) == 5
    g = {
        json.loads(ln)["unit_id"]: json.loads(ln)
        for ln in (out / "gold.jsonl").read_text().splitlines()
    }
    assert g["modal"]["gold"]["obligations"][0]["modality"] == "should"
    assert "quote" not in g["modal"]["gold"]["obligations"][0]
    assert len(g["extra"]["gold"]["obligations"]) == 2
    assert g["agree"]["source"] == "agreed"


def test_finalize_rejects_choice_of_invalid(tmp_path: Path) -> None:
    a, b, sample, units, out = _setup(tmp_path)
    gold.build(a, b, sample, units, out)
    rows = [json.loads(ln) for ln in (out / "to_adjudicate.jsonl").read_text().splitlines()]
    for r in rows:
        r["decision"] = "B"
    _w(out / "to_adjudicate.jsonl", rows)
    with pytest.raises(ValueError, match="no extraction"):
        gold.finalize(out)


def test_assert_no_gold(tmp_path: Path) -> None:
    sample = tmp_path / "s.jsonl"
    _w(sample, [{"unit_id": "g1"}])
    gold.assert_no_gold(["t1", "t2"], sample)
    with pytest.raises(ValueError, match="gold"):
        gold.assert_no_gold(["t1", "g1"], sample)


def _ex(**kw):
    from regslm.schema.obligation import Extraction

    o = _ob()
    o.update(kw)
    return Extraction.model_validate({"unit_id": "u", "obligations": [o]})


def test_tolerant_tiers() -> None:
    base = _ex()
    assert gold.tolerant_tier(base, _ex(subject="the firm")) == 0
    assert gold.tolerant_tier(base, _ex(deadline="within 72 hours")) == 0
    assert gold.tolerant_tier(base, _ex(action="report all incidents")) == 0
    assert gold.tolerant_tier(base, _ex(object="x")) == 2  # empty vs non-empty
    assert (
        gold.tolerant_tier(base, _ex(threshold={"value": 5, "unit": "%", "comparator": "<="})) == 2
    )
    assert gold.tolerant_tier(base, _ex(modality="may")) == 1
    from regslm.schema.obligation import Extraction

    assert gold.tolerant_tier(base, Extraction(unit_id="u")) == 1
    assert gold.tolerant_tier(base, None) == 1
    assert gold.tolerant_tier(Extraction(unit_id="u"), Extraction(unit_id="u")) == 0


def test_build_sorting_and_csv_roundtrip(tmp_path: Path) -> None:
    import csv as _csv

    th = "ธนาคารต้องรายงานเหตุการณ์ภายใน 72 ชั่วโมง"
    ids = ["en1", "th1", "th2"]
    sample = tmp_path / "sample.jsonl"
    _w(sample, [{"unit_id": u, "stratum": "x"} for u in ids])
    a = [_row("en1", [_ob()]), _row("th1", [_ob()]), _row("th2", [])]
    b = [_row("en1", [_ob(dl="")]), _row("th1", []), _row("th2", [_ob()])]
    _w(tmp_path / "a/labels.jsonl", a)
    _w(tmp_path / "b/labels.jsonl", b)
    units = {
        "en1": {"text": TEXT, "heading": "H", "language": "en"},
        "th1": {"text": th, "heading": "ห" * 300, "language": "th"},
        "th2": {"text": th, "heading": "H", "language": "th"},
    }
    out = tmp_path / "out"
    s = gold.build(tmp_path / "a", tmp_path / "b", sample, units, out)
    assert (s["n_tier1"], s["n_tier2"]) == (2, 1)
    rows = [json.loads(ln) for ln in (out / "to_adjudicate.jsonl").read_text().splitlines()]
    assert [r["unit_id"] for r in rows] == ["th1", "th2", "en1"]

    info = gold.export_csv(out)
    raw = (out / "adjudicate.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and info["n_rows"] == 3
    recs = list(_csv.DictReader((out / "adjudicate.csv").open(encoding="utf-8-sig", newline="")))
    assert recs[0]["text"] == th and len(recs[0]["heading"]) == 200
    assert recs[1]["A"] == gold.NO_OBLIGATION and "must" in recs[1]["B"]

    recs[0]["decision"] = "e"
    recs[0]["note"] = (
        "ธนาคาร | must | รายงานเหตุการณ์ | | cond: | thr: <= 5 % | dl: ภายใน 72 ชั่วโมง | exc: ก; ข\n"
    )
    recs[1]["decision"] = "B"
    recs[2]["decision"] = "E"
    recs[2]["note"] = "garbage"
    p = tmp_path / "filled.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    res = gold.import_csv(p, out)
    assert res["applied"] == 2 and len(res["errors"]) == 1 and res["n_undecided"] == 1
    rows = [json.loads(ln) for ln in (out / "to_adjudicate.jsonl").read_text().splitlines()]
    o = rows[0]["gold"]["obligations"][0]
    assert rows[0]["decision"] == "edit" and o["subject"] == "ธนาคาร"
    assert o["threshold"] == {"value": 5.0, "unit": "%", "comparator": "<="}
    assert o["exceptions"] == ["ก", "ข"] and o["deadline"] == "ภายใน 72 ชั่วโมง"
    assert rows[1]["decision"] == "B" and rows[1]["gold"] == rows[1]["candidate_b"]
    # no overwrite without force
    recs[1]["decision"] = "A"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs[:2])
    assert gold.import_csv(p, out)["skipped_existing"] == 2
    # E with "none"
    assert gold._edited_gold(rows[2], "none")["obligations"] == []


def test_content_rematch() -> None:
    far = {"span": {"start": 50, "end": 60}}
    a = _ex()
    b = _ex(citation={"source": "s", "unit_id": "u", **far}, deadline="within 72 hours")
    assert gold.match_obligations(a.obligations, b.obligations) == []
    assert gold.tolerant_match(a.obligations, b.obligations) == [(0, 0)]
    assert gold.tolerant_tier(a, b) == 0
    assert (
        gold.tolerant_tier(a, _ex(citation={"source": "s", "unit_id": "u", **far}, object="x")) == 2
    )
    assert (
        gold.tolerant_tier(a, _ex(citation={"source": "s", "unit_id": "u", **far}, action="appeal"))
        == 1
    )


def test_build_refuses_to_overwrite_decisions(tmp_path: Path) -> None:
    a, b, sample, units, out = _setup(tmp_path)
    gold.build(a, b, sample, units, out)
    gold.build(a, b, sample, units, out)  # no decisions yet: re-run allowed
    p = out / "to_adjudicate.jsonl"
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    rows[0]["decision"] = "A"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="1 decisions"):
        gold.build(a, b, sample, units, out)
    gold.build(a, b, sample, units, out, force=True)
