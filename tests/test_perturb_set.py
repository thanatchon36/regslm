import json
from pathlib import Path

from regslm.corpus.base import Unit
from regslm.data.perturb_set import (
    build_perturbed_test,
    map_span,
    perturb_unit_and_label,
    read_labels,
)
from regslm.rewards.normalize import extract_numbers
from regslm.schema.obligation import Extraction, Span
from regslm.schema.perturb import Change


def _unit(text: str, heading: str = "") -> Unit:
    return Unit(
        unit_id="s/d/1",
        source="s",
        doc_id="d",
        path=["1"],
        text=text,
        heading=heading,
        language="en",
        version="v",
    )


def _ex(text: str, quote: str, value: float | None, **kw: str) -> Extraction:
    i = text.index(quote)
    ob = {
        "subject": "Firms",
        "modality": "must",
        "action": "hold capital",
        **kw,
        "citation": {
            "source": "s",
            "unit_id": "s/d/1",
            "span": {"start": i, "end": i + len(quote)},
        },
    }
    if value is not None:
        ob["threshold"] = {"value": value, "unit": "%", "comparator": ">="}
    return Extraction.model_validate({"unit_id": "s/d/1", "obligations": [ob]})


def test_map_span_before_inside_after() -> None:
    ch = [Change(5, 6, "5", "12")]  # +1
    assert map_span(Span(start=10, end=20), ch) == Span(start=11, end=21)  # before
    assert map_span(Span(start=2, end=10), ch) == Span(start=2, end=11)  # inside
    assert map_span(Span(start=0, end=4), ch) == Span(start=0, end=4)  # after
    ch2 = [Change(5, 8, "100", "7")]  # -2
    assert map_span(Span(start=4, end=20), ch2) == Span(start=4, end=18)


def test_threshold_remap_and_span_grounded() -> None:
    text = "Intro 1 note. Firms must hold at least 4.5 % of exposures within 30 days."
    quote = "Firms must hold at least 4.5 % of exposures within 30 days."
    ex = _ex(text, quote, 4.5, deadline="30 days", condition="if 4.5 % applies")
    u, ex2, ch = perturb_unit_and_label(_unit(text), ex, seed=3)
    assert ex2 is not None
    o = ex2.obligations[0]
    new_pct = extract_numbers(u.text)[1]
    assert o.threshold is not None and o.threshold.value == new_pct != 4.5
    s = o.citation.span
    assert u.text[s.start : s.end].startswith("Firms must")
    assert u.text[s.start : s.end].endswith("days.")
    assert str(new_pct).rstrip("0") in o.condition or o.condition != "if 4.5 % applies"
    assert o.deadline != "30 days"
    assert any(t.field == "threshold.value" and t.original == 4.5 for t in ch.targets)


def test_eu_comma_and_thai() -> None:
    text = "Institutions shall hold 4,5 % capital."
    ex = _ex(text, text, 4.5)
    u, ex2, _ = perturb_unit_and_label(_unit(text), ex, 1)
    assert ex2 and ex2.obligations[0].threshold
    assert ex2.obligations[0].threshold.value == extract_numbers(u.text)[0] != 4.5
    th = "ธนาคารต้องดำรงเงินกองทุนร้อยละ ๒๕"
    exth = _ex(th, th, 25)
    u2, ex3, _ = perturb_unit_and_label(_unit(th), exth, 1)
    assert ex3 and ex3.obligations[0].threshold
    assert ex3.obligations[0].threshold.value == extract_numbers(u2.text)[0] != 25
    assert any(c in u2.text for c in "๐๑๒๓๔๕๖๗๘๙")


def test_no_numbers_unchanged_and_deterministic() -> None:
    text = "Firms must report promptly."
    ex = _ex(text, text, None)
    u, ex2, ch = perturb_unit_and_label(_unit(text), ex, 0)
    assert u.text == text and ex2 == ex and not ch.all
    t2 = "Firms must hold 8 % capital."
    e2 = _ex(t2, t2, 8)
    assert perturb_unit_and_label(_unit(t2), e2, 5) == perturb_unit_and_label(_unit(t2), e2, 5)
    _, none, _ = perturb_unit_and_label(_unit(t2), None, 5)
    assert none is None


def test_heading_and_writer(tmp_path: Path) -> None:
    text = "Firms must hold 8 % capital."
    u, _, ch = perturb_unit_and_label(_unit(text, "Article 92 within 30 days"), None, 2)
    assert "Article 92" in u.heading and ch.heading
    lab = tmp_path / "l.jsonl"
    ex = _ex(text, text, 8)
    lab.write_text(
        json.dumps({"unit_id": "s/d/1", "valid": True, "extraction": ex.model_dump(mode="json")})
        + "\n"
        + json.dumps({"unit_id": "s/d/2", "valid": False, "extraction": None})
        + "\n"
    )
    labels = read_labels(lab)
    assert list(labels) == ["s/d/1"]
    out = tmp_path / "o.jsonl"
    assert build_perturbed_test([_unit(text)], labels, out) == 1
    row = json.loads(out.read_text())
    assert row["memorisation_targets"][0][:2] == [0, "threshold.value"]


def test_deadline_targets() -> None:
    text = "Firms must report within 30 days or 2 business days of 8 % loss."
    ex = _ex(text, text, 8, deadline="within 30 days")
    _, ex2, ch = perturb_unit_and_label(_unit(text), ex, 1)
    assert ex2 is not None
    t = [t for t in ch.targets if t.field == "deadline.value"]
    assert len(t) == 1 and t[0].unit == "day" and t[0].original == 30 != t[0].perturbed
    assert t[0].as_row()[-1] == "day" and len(ch.targets[0].as_row()) == 4  # threshold first
