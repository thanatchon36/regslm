"""The six verifiable reward components (docs/specs/rewards.md).

Each has the signature ``(completion, unit, reference) -> float`` in [0, 1]. Components
that need a reference return 0.0 when ``reference`` is None; all return 0.0 when the
completion does not parse.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from pydantic import ValidationError

from regslm.corpus.base import Unit
from regslm.rewards.normalize import (
    deadlines_equal_relaxed,
    extract_numbers,
    match_obligations,
    normalize_text,
    number_in_text,
    thresholds_equal,
)
from regslm.schema.generation import GenExtraction, Resolved, resolve
from regslm.schema.obligation import Extraction, Obligation

GROUND_OVERLAP = 0.8
LENGTH_FACTOR = 2.0

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"\w+")

Component = Callable[[str, Unit, Extraction | None], float]


@lru_cache(maxsize=4096)
def parse(completion: str) -> GenExtraction | None:
    """Parse a completion into a GenExtraction (memoised); None on any failure. No repair."""
    s = completion.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return GenExtraction.model_validate_json(s)
    except ValidationError:
        return None


@lru_cache(maxsize=4096)
def _resolve(
    completion: str, unit_id: str, source: str, text: str, heading: str
) -> Resolved | None:
    gen = parse(completion)
    return None if gen is None else resolve(gen, unit_id, source, text, heading)


def resolved(completion: str, unit: Unit) -> Resolved | None:
    """``parse`` + ``resolve`` against the unit (memoised per (completion, unit_id))."""
    return _resolve(completion, unit.unit_id, unit.source, unit.text, unit.heading)


def schema_valid(completion: str, unit: Unit, reference: Extraction | None) -> float:
    return 1.0 if parse(completion) is not None else 0.0


def citation_resolves(completion: str, unit: Unit, reference: Extraction | None) -> float:
    """Resolved obligations / generated obligations; wrong-unit citations do not count."""
    r = resolved(completion, unit)
    if r is None:
        return 0.0
    n_gen = len(r.extraction.obligations) + len(r.unresolved)
    if n_gen == 0:
        return 1.0
    return max(0, len(r.extraction.obligations) - r.wrong_unit) / n_gen


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(s))


def _fields(o: Obligation) -> list[tuple[str, str]]:
    raw = [
        ("subject", o.subject),
        ("action", o.action),
        ("object", o.object),
        ("condition", o.condition),
        ("deadline", o.deadline),
        *[("exception", e) for e in o.exceptions],
    ]
    return [(k, v) for k, v in raw if v.strip()]


def _obligation_grounded(o: Obligation, unit: Unit, text_n: str, heading_n: str) -> float:
    fields = _fields(o)  # subject is mandatory, so never empty
    span = o.citation.span
    sentence = unit.text[span.start : span.end] if span.end <= len(unit.text) else ""
    sent_tokens = set(_tokens(sentence))
    ok = 0
    for kind, value in fields:
        art = kind in ("subject", "object")
        n = normalize_text(value, strip_articles=art)
        if n and (n in text_n or (kind == "subject" and heading_n and n in heading_n)):
            ok += 1
            continue
        toks = _tokens(n)
        if toks and sum(t in sent_tokens for t in toks) / len(toks) >= GROUND_OVERLAP:
            ok += 1
    return ok / len(fields)


def span_grounded(completion: str, unit: Unit, reference: Extraction | None) -> float:
    r = resolved(completion, unit)
    if r is None:
        return 0.0
    ext = r.extraction
    if reference is not None and len(completion) > LENGTH_FACTOR * len(reference.model_dump_json()):
        return 0.0  # length guard: no dumping the unit into fields
    if not ext.obligations:
        return 0.0 if r.unresolved else 1.0
    text_n = normalize_text(unit.text)
    heading_n = normalize_text(unit.heading)
    scores = [_obligation_grounded(o, unit, text_n, heading_n) for o in ext.obligations]
    return sum(scores) / len(scores)


def numeric_exact(completion: str, unit: Unit, reference: Extraction | None) -> float:
    r = resolved(completion, unit)
    if r is None or reference is None:
        return 0.0
    ext = r.extraction
    pairs = match_obligations(ext.obligations, reference.obligations)
    if not pairs:
        return 0.0 if any(g.threshold for g in reference.obligations) else 1.0
    text_numbers = extract_numbers(unit.text)
    ok = 0
    for i, j in pairs:
        p, g = ext.obligations[i].threshold, reference.obligations[j].threshold
        if p is not None and not number_in_text(p.value, text_numbers):
            continue
        ok += thresholds_equal(p, g)
    return ok / len(pairs)


def deadline_exact(completion: str, unit: Unit, reference: Extraction | None) -> float:
    r = resolved(completion, unit)
    if r is None or reference is None:
        return 0.0
    ext = r.extraction
    pairs = match_obligations(ext.obligations, reference.obligations)
    if not pairs:
        return 0.0 if any(g.deadline.strip() for g in reference.obligations) else 1.0
    ok = sum(
        deadlines_equal_relaxed(ext.obligations[i].deadline, reference.obligations[j].deadline)
        for i, j in pairs
    )
    return ok / len(pairs)


def no_hallucinated_obligation(completion: str, unit: Unit, reference: Extraction | None) -> float:
    r = resolved(completion, unit)
    if r is None or reference is None:
        return 0.0
    obs = r.extraction.obligations
    n_pred = len(obs) + len(r.unresolved)
    fp = len(obs) - len(match_obligations(obs, reference.obligations)) + len(r.unresolved)
    return 1.0 - fp / max(1, n_pred)


COMPONENTS: dict[str, Component] = {
    "schema": schema_valid,
    "cite": citation_resolves,
    "ground": span_grounded,
    "num": numeric_exact,
    "dl": deadline_exact,
    "hall": no_hallucinated_obligation,
}
