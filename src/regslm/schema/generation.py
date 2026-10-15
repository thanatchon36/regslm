"""What models *generate*, and how it becomes an ``Extraction`` (paper §3.2, D-020).

LLMs cannot count characters: in the teacher pilot only 49 % of model-written
``citation.span`` offsets covered the trigger (results/runs/teacher-bench-label-seqs24).
Models therefore emit a verbatim ``quote`` of the sentence containing the trigger, and
``resolve`` maps it to character offsets deterministically:

1. exact substring match (first occurrence);
2. match after whitespace/quote-mark normalisation, mapped back to original offsets;
3. fuzzy alignment (``difflib`` matching blocks): ≥ ``FUZZY_MIN`` of the quote covered, and
   the aligned window ≤ 1.3 × the quote length.

An obligation whose quote does not resolve is *ungrounded*: it is dropped from the
``Extraction`` and counted, so evaluation can score it as a false positive and the
``citation_resolves`` reward can penalise it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from regslm.schema.obligation import (
    Citation,
    Extraction,
    Modality,
    Obligation,
    Span,
    Threshold,
)

FUZZY_MIN = 0.85
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class GenCitation(_Strict):
    source: Annotated[str, Field(min_length=1)]
    unit_id: Annotated[str, Field(min_length=1)]
    quote: Annotated[str, Field(min_length=1, max_length=2000)]


class GenObligation(_Strict):
    subject: Annotated[str, Field(min_length=1, max_length=300)]
    modality: Modality
    action: Annotated[str, Field(min_length=1, max_length=300)]
    object: Annotated[str, Field(max_length=500)] = ""
    condition: Annotated[str, Field(max_length=800)] = ""
    threshold: Threshold | None = None
    deadline: Annotated[str, Field(max_length=200)] = ""
    exceptions: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list)
    citation: GenCitation
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class GenExtraction(_Strict):
    """Generation target. ``rationale`` first so it conditions the labels (D-016)."""

    unit_id: str
    rationale: str | None = None
    obligations: list[GenObligation] = Field(default_factory=list)


@dataclass
class Resolved:
    extraction: Extraction
    unresolved: list[GenObligation] = field(default_factory=list)
    wrong_unit: int = 0  # quote resolved but citation.unit_id/source did not match the unit


def _norm_with_map(s: str) -> tuple[str, list[int]]:
    """Collapse whitespace, unify quote marks, lowercase; keep a map to original offsets."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(s.translate(_QUOTES)):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
            continue
        out.append(ch.lower())
        idx.append(i)
        prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def resolve_span(text: str, quote: str) -> Span | None:
    """Character span of ``quote`` in ``text`` (see module docstring), or None."""
    q = quote.strip()
    if not q or not text:
        return None
    i = text.find(q)
    if i >= 0:
        return Span(start=i, end=i + len(q))
    nt, tmap = _norm_with_map(text)
    nq, _ = _norm_with_map(q)
    j = nt.find(nq) if nq else -1
    if j >= 0:
        return Span(start=tmap[j], end=tmap[j + len(nq) - 1] + 1)
    # Fuzzy: align quote to text with difflib matching blocks; accept if enough of the quote
    # is covered and the aligned window is not much longer than the quote.
    sm = difflib.SequenceMatcher(None, nq, nt, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 3]
    if blocks:
        covered = sum(b.size for b in blocks)
        lo, hi = blocks[0].b, blocks[-1].b + blocks[-1].size
        if covered / len(nq) >= FUZZY_MIN and (hi - lo) <= 1.3 * len(nq):
            return Span(start=tmap[lo], end=tmap[hi - 1] + 1)
    return None


def resolve(gen: GenExtraction, unit_id: str, source: str, text: str) -> Resolved:
    """Turn a generated extraction into an ``Extraction`` with resolved character spans."""
    obligations: list[Obligation] = []
    unresolved: list[GenObligation] = []
    wrong_unit = 0
    for g in gen.obligations:
        span = resolve_span(text, g.citation.quote)
        if span is None:
            unresolved.append(g)
            continue
        if g.citation.unit_id != unit_id or g.citation.source != source:
            wrong_unit += 1
        data = g.model_dump(exclude={"citation"})
        obligations.append(
            Obligation(
                **data,
                citation=Citation(source=source, unit_id=unit_id, span=span),
            )
        )
    ex = Extraction(unit_id=unit_id, rationale=gen.rationale, obligations=obligations)
    return Resolved(extraction=ex, unresolved=unresolved, wrong_unit=wrong_unit)
