import pytest

from regslm.schema import GenExtraction, resolve, resolve_span

TEXT = (
    "Subject to Articles 93 and 94, institutions shall at all times satisfy  the following "
    "own funds requirements: a Common Equity Tier 1 capital ratio of 4,5 %; a Tier 1 "
    "capital ratio of 6 %."
)


@pytest.mark.parametrize(
    "quote",
    [
        "institutions shall at all times satisfy  the following own funds requirements:",
        "institutions shall at all times satisfy the following own funds requirements:",  # ws
        "Institutions Shall At All Times satisfy the following own funds requirements",  # case
        "institutions shall at all time satisfy the following own fund requirements",  # fuzzy
    ],
)
def test_resolve_span_variants(quote: str) -> None:
    s = resolve_span(TEXT, quote)
    assert s is not None
    assert "shall at all times" in TEXT[s.start : s.end]


def test_resolve_span_rejects_unrelated() -> None:
    assert resolve_span(TEXT, "A covered company must file the report within 30 days.") is None
    assert resolve_span(TEXT, "") is None


def test_resolve_drops_unresolved_and_counts_wrong_unit() -> None:
    gen = GenExtraction.model_validate(
        {
            "unit_id": "u1",
            "obligations": [
                {
                    "subject": "institutions",
                    "modality": "must",
                    "action": "satisfy",
                    "threshold": {"value": 4.5, "unit": "%", "comparator": ">="},
                    "citation": {"source": "eurlex", "unit_id": "WRONG", "quote": "a Tier 1 capital ratio of 6 %"},
                },
                {
                    "subject": "banks",
                    "modality": "must",
                    "action": "report",
                    "citation": {"source": "eurlex", "unit_id": "u1", "quote": "Banks must report quarterly."},
                },
            ],
        }
    )
    r = resolve(gen, unit_id="u1", source="eurlex", text=TEXT)
    assert len(r.extraction.obligations) == 1 and len(r.unresolved) == 1
    assert r.wrong_unit == 1
    ob = r.extraction.obligations[0]
    assert ob.citation.unit_id == "u1"
    assert TEXT[ob.citation.span.start : ob.citation.span.end] == "a Tier 1 capital ratio of 6 %"
