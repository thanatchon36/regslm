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
                    "citation": {
                        "source": "eurlex",
                        "unit_id": "WRONG",
                        "quote": "a Tier 1 capital ratio of 6 %",
                    },
                },
                {
                    "subject": "banks",
                    "modality": "must",
                    "action": "report",
                    "citation": {
                        "source": "eurlex",
                        "unit_id": "u1",
                        "quote": "Banks must report quarterly.",
                    },
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


def test_resolve_quote_from_heading_maps_to_whole_point() -> None:
    gen = GenExtraction.model_validate(
        {
            "unit_id": "u2",
            "obligations": [
                {
                    "subject": "a bank",
                    "modality": "must",
                    "action": "appoint",
                    "object": "a chief risk officer",
                    "citation": {"source": "s", "unit_id": "u2", "quote": "A bank must:"},
                }
            ],
        }
    )
    point = "(b) appoint a chief risk officer;"
    r = resolve(gen, "u2", "s", point, heading="Governance | A bank must:")
    assert r.unresolved == []
    span = r.extraction.obligations[0].citation.span
    assert (span.start, span.end) == (0, len(point))
    assert resolve(gen, "u2", "s", point).unresolved  # no heading → unresolved
