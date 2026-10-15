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
