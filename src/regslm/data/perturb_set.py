"""Numerically perturbed test set (paper §4.3 / §3.5): perturb unit text, remap the label."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from regslm.corpus.base import Unit
from regslm.rewards.normalize import deadline_pairs, extract_numbers
from regslm.schema.obligation import Citation, Extraction, Obligation, Span, Threshold
from regslm.schema.perturb import Change, perturb_numbers

_FIELDS = ("subject", "action", "object", "condition", "deadline")


@dataclass(frozen=True)
class MemTarget:
    obligation: int
    field: str
    original: float
    perturbed: float
    unit: str = ""  # canonical time unit, for ``deadline.value`` targets

    def as_row(self) -> list[Any]:
        """Serialised form: ``[obligation, field, original, perturbed]`` (+ unit for deadlines)."""
        row: list[Any] = [self.obligation, self.field, self.original, self.perturbed]
        return [*row, self.unit] if self.unit else row


@dataclass(frozen=True)
class Changes:
    text: list[Change]
    heading: list[Change]
    targets: list[MemTarget] = field(default_factory=list)

    @property
    def all(self) -> list[Change]:
        return [*self.text, *self.heading]


def map_span(span: Span, changes: list[Change]) -> Span:
    """Shift a span through an edit list (offsets refer to the original text)."""
    start = span.start + sum(
        len(c.perturbed) - len(c.original) for c in changes if c.end <= span.start
    )
    end = span.end + sum(len(c.perturbed) - len(c.original) for c in changes if c.end <= span.end)
    return Span(start=start, end=max(end, start + 1))


def _replace_all(s: str, mapping: dict[str, str]) -> str:
    if not s or not mapping:
        return s
    keys = sorted(mapping, key=len, reverse=True)
    pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)")
    return pat.sub(lambda m: mapping[m.group(1)], s)


def _single_number(s: str) -> float | None:
    nums = extract_numbers(s)
    return nums[0] if len(nums) == 1 else None


def remap_extraction(ex: Extraction, changes: Changes) -> tuple[Extraction, list[MemTarget]]:
    mapping: dict[str, str] = {}
    for c in changes.all:
        mapping.setdefault(c.original, c.perturbed)
    numeric = [
        (v, w)
        for c in changes.all
        if (v := _single_number(c.original)) is not None
        and (w := _single_number(c.perturbed)) is not None
    ]
    obls: list[Obligation] = []
    targets: list[MemTarget] = []
    for i, o in enumerate(ex.obligations):
        upd: dict[str, Any] = {f: _replace_all(getattr(o, f), mapping) for f in _FIELDS}
        upd["exceptions"] = [_replace_all(e, mapping) for e in o.exceptions]
        upd["citation"] = Citation(
            source=o.citation.source,
            unit_id=o.citation.unit_id,
            span=map_span(o.citation.span, changes.text),
        )
        if o.threshold is not None:
            for v, w in numeric:
                if math.isclose(o.threshold.value, v, rel_tol=1e-9, abs_tol=1e-12):
                    upd["threshold"] = Threshold(
                        value=w, unit=o.threshold.unit, comparator=o.threshold.comparator
                    )
                    targets.append(MemTarget(i, "threshold.value", v, w))
                    break
        if o.deadline:
            old_p, new_p = deadline_pairs(o.deadline), deadline_pairs(upd["deadline"])
            for unit in sorted({u for _, u in old_p}):
                olds = [n for n, u in old_p if u == unit]
                news = [n for n, u in new_p if u == unit]
                if len(olds) == 1 and len(news) == 1 and olds[0] != news[0]:
                    targets.append(MemTarget(i, "deadline.value", olds[0], news[0], unit))
        obls.append(o.model_copy(update=upd))
    return ex.model_copy(update={"obligations": obls}), targets


def perturb_unit_and_label(
    unit: Unit, ex: Extraction | None, seed: int = 0
) -> tuple[Unit, Extraction | None, Changes]:
    text, tc = perturb_numbers(unit.text, seed)
    heading, hc = perturb_numbers(unit.heading, seed) if unit.heading else (unit.heading, [])
    changes = Changes(tc, hc)
    new_unit = unit.model_copy(update={"text": text, "heading": heading})
    if ex is None:
        return new_unit, None, changes
    new_ex, targets = remap_extraction(ex, changes)
    return new_unit, new_ex, replace(changes, targets=targets)


def read_labels(path: Path) -> dict[str, Extraction]:
    """Valid extractions from a teacher ``labels.jsonl`` (or bare Extraction rows)."""
    out: dict[str, Extraction] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        if "extraction" in row:
            if not row.get("valid") or not row["extraction"]:
                continue
            row = row["extraction"]
        out[row["unit_id"]] = Extraction.model_validate(row)
    return out


def build_perturbed_test(
    units: Iterable[Unit], labels: dict[str, Extraction], out: Path, seed: int = 0
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for u in units:
            ex = labels.get(u.unit_id)
            if ex is None:
                continue
            nu, nex, ch = perturb_unit_and_label(u, ex, seed)
            assert nex is not None
            row = {
                "unit": nu.model_dump(mode="json"),
                "label": nex.model_dump(mode="json"),
                "changes": {
                    "text": [vars(c) for c in ch.text],
                    "heading": [vars(c) for c in ch.heading],
                },
                "memorisation_targets": [t.as_row() for t in ch.targets],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
