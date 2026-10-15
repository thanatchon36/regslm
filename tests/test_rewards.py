"""Acceptance tests for regslm.rewards (docs/specs/rewards.md)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from regslm.corpus.base import Unit
from regslm.rewards import (
    batch_stats,
    citation_resolves,
    deadline_exact,
    load_weights,
    match_obligations,
    no_hallucinated_obligation,
    normalize_text,
    numeric_exact,
    parse,
    reward,
    schema_valid,
    span_grounded,
)
from regslm.rewards.normalize import (
    deadline_pairs,
    deadlines_equal_relaxed,
    digits_to_ascii,
    extract_numbers,
    hungarian,
    normalize_unit,
    span_iou,
    thresholds_equal,
)
from regslm.schema.obligation import Extraction, Obligation, Span, Threshold

SENT1 = "Banks must hold capital of at least 8 percent of risk-weighted assets within 30 days."
SENT2 = "A bank may not pay dividends."
TEXT = f"{SENT1} {SENT2}"
UNIT = Unit(
    unit_id="us/12cfr/217/10",
    source="us",
    doc_id="12cfr217",
    path=["217.10"],
    heading="Capital requirements",
    text=TEXT,
    language="en",
    version="2024-01-01",
)


def ob(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "subject": "Banks",
        "modality": "must",
        "action": "hold capital",
        "object": "risk-weighted assets",
        "threshold": {"value": 8, "unit": "%", "comparator": ">="},
        "deadline": "within 30 days",
        "citation": {
            "source": "us",
            "unit_id": UNIT.unit_id,
            "span": {"start": 0, "end": len(SENT1)},
        },
    }
    base.update(kw)
    return base


def ext(*obs: dict[str, Any]) -> dict[str, Any]:
    return {"unit_id": UNIT.unit_id, "obligations": list(obs)}


def js(d: dict[str, Any]) -> str:
    return json.dumps(d)


def ref(*obs: dict[str, Any]) -> Extraction:
    return Extraction.model_validate(ext(*obs))


GOLD = ref(ob())
PERFECT = js(ext(ob()))
INVALID = "not json"
EMPTY = js(ext())
HALLUC = js(
    ext(
        ob(),
        ob(
            subject="Regulators",
            modality="should",
            action="audit",
            object="",
            threshold=None,
            deadline="",
            citation={"source": "us", "unit_id": UNIT.unit_id, "span": {"start": 50, "end": 80}},
        ),
    )
)
WRONG_SPAN = js(
    ext(ob(citation={"source": "us", "unit_id": UNIT.unit_id, "span": {"start": 0, "end": 9999}}))
)
PARAPHRASE = js(
    ext(ob(subject="Lenders", action="keep reserves", object="safe holdings", deadline="soon"))
)
MEMORISED = js(ext(ob(threshold={"value": 9, "unit": "%", "comparator": ">="})))


def test_parse_and_fence() -> None:
    assert parse(PERFECT) is not None
    assert parse(f"```json\n{PERFECT}\n```") == parse(PERFECT)
    assert parse(INVALID) is None
    assert parse(js({"unit_id": "x", "bogus": 1})) is None


def test_schema_valid() -> None:
    assert schema_valid(PERFECT, UNIT, GOLD) == 1.0
    assert schema_valid(INVALID, UNIT, GOLD) == 0.0
    assert schema_valid(EMPTY, UNIT, GOLD) == 1.0


def test_citation_resolves() -> None:
    assert citation_resolves(PERFECT, UNIT, None) == 1.0
    assert citation_resolves(INVALID, UNIT, None) == 0.0
    assert citation_resolves(EMPTY, UNIT, None) == 1.0
    assert citation_resolves(WRONG_SPAN, UNIT, None) == 0.0
    assert citation_resolves(HALLUC, UNIT, None) == 1.0
    other = js(
        ext(ob(citation={"source": "eu", "unit_id": UNIT.unit_id, "span": {"start": 0, "end": 5}}))
    )
    assert citation_resolves(other, UNIT, None) == 0.0


def test_span_grounded() -> None:
    assert span_grounded(PERFECT, UNIT, GOLD) == 1.0
    assert span_grounded(INVALID, UNIT, GOLD) == 0.0
    assert span_grounded(EMPTY, UNIT, GOLD) == 1.0
    assert span_grounded(PARAPHRASE, UNIT, GOLD) < 0.5
    # subject inherited from heading
    heading = js(ext(ob(subject="Capital requirements")))
    assert span_grounded(heading, UNIT, None) == 1.0
    # token overlap with cited sentence tolerates reordering
    reorder = js(ext(ob(action="capital hold")))
    assert span_grounded(reorder, UNIT, None) == 1.0
    # wrong span: out-of-range citation gives no sentence for overlap
    assert span_grounded(WRONG_SPAN.replace("hold capital", "keep reserves"), UNIT, None) < 1.0
    # article stripped for subject
    assert span_grounded(js(ext(ob(subject="The banks"))), UNIT, None) == 1.0


def test_length_guard() -> None:
    padded = js(ext(ob(condition=SENT1 * 8)))
    assert span_grounded(padded, UNIT, GOLD) == 0.0
    assert span_grounded(padded, UNIT, None) == 1.0


def test_numeric_exact() -> None:
    assert numeric_exact(PERFECT, UNIT, GOLD) == 1.0
    assert numeric_exact(INVALID, UNIT, GOLD) == 0.0
    assert numeric_exact(PERFECT, UNIT, None) == 0.0
    assert numeric_exact(EMPTY, UNIT, GOLD) == 0.0
    assert numeric_exact(EMPTY, UNIT, ref()) == 1.0
    assert numeric_exact(EMPTY, UNIT, ref(ob(threshold=None))) == 1.0
    assert numeric_exact(MEMORISED, UNIT, ref(ob(threshold={"value": 9, "unit": "%"}))) == 0.0
    assert numeric_exact(MEMORISED, UNIT, GOLD) == 0.0
    aliased = js(ext(ob(threshold={"value": 8, "unit": "per cent", "comparator": ">="})))
    assert numeric_exact(aliased, UNIT, GOLD) == 1.0
    wrong_cmp = js(ext(ob(threshold={"value": 8, "unit": "%", "comparator": "<="})))
    assert numeric_exact(wrong_cmp, UNIT, GOLD) == 0.0


def test_deadline_exact() -> None:
    assert deadline_exact(PERFECT, UNIT, GOLD) == 1.0
    assert deadline_exact(INVALID, UNIT, GOLD) == 0.0
    assert deadline_exact(PERFECT, UNIT, None) == 0.0
    assert deadline_exact(EMPTY, UNIT, GOLD) == 0.0
    assert deadline_exact(EMPTY, UNIT, ref()) == 1.0
    assert deadline_exact(EMPTY, UNIT, ref(ob(deadline=""))) == 1.0
    assert deadline_exact(js(ext(ob(deadline="no later than 30 days"))), UNIT, GOLD) == 1.0
    assert deadline_exact(js(ext(ob(deadline="within 60 days"))), UNIT, GOLD) == 0.0
    assert deadline_exact(PARAPHRASE, UNIT, GOLD) == 0.0


def test_no_hallucinated() -> None:
    assert no_hallucinated_obligation(PERFECT, UNIT, GOLD) == 1.0
    assert no_hallucinated_obligation(INVALID, UNIT, GOLD) == 0.0
    assert no_hallucinated_obligation(PERFECT, UNIT, None) == 0.0
    assert no_hallucinated_obligation(HALLUC, UNIT, GOLD) == 0.5
    assert no_hallucinated_obligation(EMPTY, UNIT, ref()) == 1.0
    assert no_hallucinated_obligation(WRONG_SPAN, UNIT, GOLD) == 0.0


def test_composite_empty_vs_correct() -> None:
    good = reward(PERFECT, UNIT, GOLD)
    empty = reward(EMPTY, UNIT, GOLD)
    assert good.total > empty.total
    assert good.recall == 1.0 and good.recall_term == 1.5
    assert empty.recall == 0.0 and empty.empty
    assert reward(INVALID, UNIT, GOLD).total == 0.0
    assert good.total == pytest.approx(1.0 + 1.5)
    assert reward(HALLUC, UNIT, GOLD).total < good.total


def test_composite_reference_free_and_weights(tmp_path: Path) -> None:
    free = reward(PERFECT, UNIT, None)
    assert free.total == 1.0 and free.recall_term == 0.0
    p = tmp_path / "w.yaml"
    p.write_text("rec: 0\ndl: 2\n")
    assert load_weights(p)["dl"] == 2.0
    assert reward(PERFECT, UNIT, GOLD, p).total == pytest.approx(1.0)
    assert load_weights(None) == load_weights({})
    p.write_text("")
    assert load_weights(p)["rec"] == 1.5
    with pytest.raises(ValueError):
        load_weights({"nope": 1})
    zero = reward(PERFECT, UNIT, None, {"cite": 0, "ground": 0})
    assert zero.total == 0.0


def test_default_yaml_matches_defaults() -> None:
    cfg = Path(__file__).parents[1] / "config" / "rewards.yaml"
    assert load_weights(cfg) == load_weights({})


def test_batch_stats() -> None:
    bs = [reward(PERFECT, UNIT, GOLD), reward(EMPTY, UNIT, GOLD)]
    st = batch_stats([PERFECT, EMPTY], bs)
    assert st["reward/empty_frac"] == 0.5
    assert st["reward/recall"] == 0.5
    assert st["reward/mean_len"] == (len(PERFECT) + len(EMPTY)) / 2
    assert batch_stats([], [])["reward/total"] == 0.0


def test_normalize() -> None:
    assert normalize_text("  The  Bank. ", strip_articles=True) == "bank"
    assert normalize_text("The Bank") == "the bank"
    assert normalize_text("An ＡＢＣ;") == "an abc"
    assert normalize_text("Anchor", strip_articles=True) == "anchor"
    assert normalize_unit("ร้อยละ") == normalize_unit("Per cent") == "%"
    assert digits_to_ascii("๑,๐๐๐") == "1000"
    assert extract_numbers("1,000 and 2.5 and ๓") == [1000.0, 2.5, 3.0]
    assert deadline_pairs("within thirty days") == {(30.0, "day")}
    assert deadline_pairs("๓๐ วัน") == {(30.0, "day")}
    assert deadlines_equal_relaxed("Promptly.", "promptly")
    assert not deadlines_equal_relaxed("promptly", "")
    assert thresholds_equal(None, None)
    assert not thresholds_equal(Threshold(value=1), None)


def _o(start: int, end: int, modality: str = "must") -> Obligation:
    return Obligation.model_validate(
        ob(
            modality=modality,
            citation={"source": "us", "unit_id": "u", "span": {"start": start, "end": end}},
        )
    )


def test_matching_and_hungarian() -> None:
    assert span_iou(Span(start=0, end=10), Span(start=5, end=15)) == pytest.approx(1 / 3)
    # greedy would take pred0->gold0 (iou 0.6) and strand pred1; optimum matches both
    pred = [_o(0, 10), _o(3, 13)]
    gold = [_o(0, 10), _o(0, 10, "may"), _o(4, 14)]
    assert match_obligations(pred, gold) == [(0, 0), (1, 2)]
    assert match_obligations([], gold) == []
    assert match_obligations([_o(0, 10)], [_o(0, 10, "may")]) == []
    assert match_obligations([_o(0, 10), _o(0, 10)], [_o(0, 10)]) == [(0, 0)]
    assert hungarian([]) == []
    assert sorted(hungarian([[4, 1], [2, 0], [3, 3]])) == [(0, 1), (1, 0)]


def test_extract_numbers_eu_formats() -> None:
    from regslm.rewards.normalize import extract_numbers

    assert 4.5 in extract_numbers("ratio of 4,5 %")
    assert 5_000_000.0 in extract_numbers("EUR 5 000 000")
    assert 1000.0 in extract_numbers("1,000 days")
