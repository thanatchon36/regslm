import json

import pytest
from pydantic import ValidationError

from regslm.schema import Citation, Extraction, Modality, Obligation, Span, Threshold, json_schema
from regslm.schema.perturb import perturb_numbers


def _ob(**kw: object) -> Obligation:
    base: dict[str, object] = {
        "subject": "institutions",
        "modality": "must",
        "action": "satisfy",
        "citation": {"source": "eurlex", "unit_id": "eurlex/CRR/92/1/a", "span": {"start": 0, "end": 10}},
    }
    base.update(kw)
    return Obligation.model_validate(base)


def test_roundtrip() -> None:
    ob = _ob(threshold={"value": 4.5, "unit": "%", "comparator": ">="})
    ex = Extraction(unit_id="u", obligations=[ob])
    assert Extraction.model_validate_json(ex.model_dump_json()) == ex
    assert ob.modality is Modality.MUST
    assert ob.threshold == Threshold(value=4.5, unit="%")


def test_empty_extraction_is_valid() -> None:
    assert Extraction.model_validate({"unit_id": "u"}).obligations == []


@pytest.mark.parametrize(
    "bad",
    [
        {"modality": "shall"},
        {"subject": ""},
        {"confidence": 1.5},
        {"extra_field": 1},
        {"citation": {"source": "s", "unit_id": "u", "span": {"start": 5, "end": 5}}},
    ],
)
def test_rejects_invalid(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _ob(**bad)


def test_span_order() -> None:
    with pytest.raises(ValidationError):
        Span(start=3, end=1)
    assert Citation(source="s", unit_id="u", span=Span(start=0, end=1)).span.end == 1


def test_json_schema_exports() -> None:
    schema = json.loads(json_schema())
    assert schema["title"] == "Extraction"
    assert "Obligation" in schema["$defs"]


@pytest.mark.parametrize(
    ("text", "originals"),
    [
        ("Institutions shall maintain a ratio of 4.5 % at all times.", ["4.5"]),
        ("file within 30 days after quarter end", ["30"]),
        ("assets exceeding $250 billion", ["250"]),
        ("by 31 March 2025", ["31 March 2025"]),
        ("ไม่เกินร้อยละ ๒๕ ของเงินกองทุน", ["๒๕"]),
        ("within 1,000 days", ["1,000"]),
    ],
)
def test_perturb_changes_quantities(text: str, originals: list[str]) -> None:
    out, changes = perturb_numbers(text, seed=1)
    assert [c.original for c in changes] == originals
    for c in changes:
        assert c.perturbed != c.original
        assert c.perturbed in out
    assert out != text


@pytest.mark.parametrize(
    "text",
    ["in accordance with Article 92(1)", "see § 217.10 and 12 CFR part 3", "(a) the firm must"],
)
def test_perturb_leaves_references(text: str) -> None:
    out, changes = perturb_numbers(text, seed=1)
    assert out == text and changes == []


def test_perturb_deterministic_and_seeded() -> None:
    t = "a minimum of 8 % within 10 business days"
    assert perturb_numbers(t, 3) == perturb_numbers(t, 3)
    assert perturb_numbers(t, 3)[0] != t


def test_perturb_thai_digits_stay_thai() -> None:
    _out, ch = perturb_numbers("ภายใน ๓๐ วัน", seed=0)
    assert all(c in "๐๑๒๓๔๕๖๗๘๙" for c in ch[0].perturbed)
