"""Needle-in-rulebook contexts (paper §4.6): one target unit inserted into real filler units.

``build_context`` produces exactly ``target_tokens`` tokens (as counted by the supplied
tokenizer) of: filler units from one held-out document, with the target unit block inserted at
``depth`` (fraction of the filler length). The tokenizer is any object with
``encode(str) -> list[int]`` and ``decode(list[int]) -> str`` (a fake one in tests).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from regslm.corpus.base import Unit
from regslm.teacher.prompts import user_prompt

LENGTHS = (4096, 8192, 16384)
DEPTHS = (0.1, 0.5, 0.9)

NEEDLE_SYSTEM = (
    "You are given a long excerpt of a rulebook made of many units. Extract the legal "
    "obligations of ONE unit only, as JSON with keys unit_id, rationale, obligations "
    "(subject, modality, action, object, condition, threshold, deadline, exceptions, "
    "citation{source, unit_id, quote}, confidence). Ignore every other unit."
)


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass
class Needle:
    target_unit_id: str
    target_tokens: int
    depth: float
    context: str
    n_tokens: int  # exact token count of ``context``
    needle_start_token: int
    messages: list[dict[str, str]]


def unit_block(u: Unit) -> str:
    return user_prompt(u.source, u.unit_id, u.heading, u.text) + "\n\n"


def question(unit_id: str) -> str:
    return f"\n\nTask: extract the obligations of unit_id {unit_id} only."


def build_context(
    filler: Sequence[Unit],
    target: Unit,
    tok: Tokenizer,
    target_tokens: int,
    depth: float,
) -> Needle:
    """Exactly ``target_tokens`` tokens; raises ValueError if the filler is too short."""
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1]")
    needle_ids = tok.encode(unit_block(target))
    n_fill = target_tokens - len(needle_ids)
    if n_fill < 0:
        raise ValueError("target unit longer than the context")
    fill_ids: list[int] = []
    for u in filler:
        if u.unit_id == target.unit_id:
            continue
        fill_ids += tok.encode(unit_block(u))
        if len(fill_ids) >= n_fill:
            break
    if len(fill_ids) < n_fill:
        raise ValueError(f"filler has {len(fill_ids)} tokens, need {n_fill}")
    fill_ids = fill_ids[:n_fill]
    at = round(depth * n_fill)
    ids = fill_ids[:at] + needle_ids + fill_ids[at:]
    ctx = tok.decode(ids)
    return Needle(
        target_unit_id=target.unit_id,
        target_tokens=target_tokens,
        depth=depth,
        context=ctx,
        n_tokens=len(ids),
        needle_start_token=at,
        messages=[
            {"role": "system", "content": NEEDLE_SYSTEM},
            {"role": "user", "content": ctx + question(target.unit_id)},
        ],
    )


def pick_document(units: Sequence[Unit], min_filler_tokens: int, tok: Tokenizer) -> list[Unit]:
    """Units of the largest document (stable order of appearance) that is long enough."""
    docs: dict[str, list[Unit]] = {}
    for u in units:
        docs.setdefault(u.doc_id, []).append(u)
    for _, us in sorted(docs.items(), key=lambda kv: -len(kv[1])):
        if sum(len(tok.encode(unit_block(u))) for u in us) >= min_filler_tokens:
            return us
    raise ValueError("no document has enough text for the requested context length")


def score_needle(completion: str, target: Unit, reference: dict[str, Any]) -> dict[str, Any]:
    """OF1 on the target unit + number of obligations attributed to other units."""
    from regslm.eval import metrics as M
    from regslm.rewards.components import parse, resolved
    from regslm.schema.obligation import Extraction

    ref = Extraction.model_validate(reference)
    gen = parse(completion)
    if gen is None:
        res = M.UnitResult(target.unit_id, ref, None)
        return {"valid": False, "of1": float(M.summarize(M.per_unit_counts([res]))["of1"]),
                "n_other_unit": 0}  # fmt: skip
    other = sum(
        1
        for o in gen.obligations
        if o.citation.unit_id != target.unit_id or gen.unit_id != target.unit_id
    )
    r = resolved(completion, target)
    pred = r.extraction if r is not None else None
    n_unres = len(r.unresolved) if r is not None else 0
    df = M.per_unit_counts([M.UnitResult(target.unit_id, ref, pred, n_unres)])
    return {"valid": True, "of1": float(M.summarize(df)["of1"]), "n_other_unit": other}
