"""Numeric perturbation for the contamination control (paper §4.3 "Memorisation").

``perturb_numbers(text, seed)`` rewrites every threshold-like quantity in a unit —
percentages, currency amounts, day/month/year counts, ratios and calendar dates — to a
different plausible value, and returns the mapping ``original -> perturbed``. A model
that outputs an original value on the perturbed text is reading from memory, not from
the input; that is the *memorisation rate*.

Design (docs/DECISIONS.md D-009):
- Deterministic given ``seed`` and text, so the perturbed test set is reproducible.
- Values change by a factor drawn from [0.55, 0.85] ∪ [1.2, 1.8] (never identity, never a
  round-trip to another value in the same unit), keeping the same number of decimals so
  the text still reads naturally.
- Dates shift by 3–11 days and the year by ±1, keeping the original format.
- Section/article numbers, cross-references ("Article 92(1)", "§ 217.10", "12 CFR") and
  list markers are **not** perturbed — only quantities.
- Thai digits (๐-๙) are perturbed and written back in Thai digits.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_THAI = "๐๑๒๓๔๕๖๗๘๙"
_TO_ASCII = str.maketrans(_THAI, "0123456789")
_TO_THAI = str.maketrans("0123456789", _THAI)

# EU texts use decimal commas ("4,5 %") and space-grouped thousands ("EUR 5 000 000").
_D = "[0-9๐-๙]"
_NUM = (
    rf"{_D}{{1,3}}(?:[,\u00a0 ]{_D}{{3}})+(?:[.,]{_D}+)?(?!{_D})"  # 1,000 / 5 000 000
    rf"|{_D}+[.,]{_D}{{1,2}}(?!{_D})"  # 4.5 / 4,5
    rf"|{_D}+(?:\.{_D}+)?"
)
_UNIT = (
    r"(?:\s?(?:%|per\s?cent|percent|basis points|bps|times|"
    r"(?:business |calendar |working )?(?:days?|months?|years?|weeks?|hours?)|"
    r"(?:(?:million|billion|thousand) )?(?:USD|EUR|GBP|THB|HKD|SGD|euros?|dollars|baht)|"
    r"วัน|เดือน|ปี|บาท|ล้านบาท|เท่า))"
)
_CURR = r"(?:USD|EUR|GBP|THB|HKD|SGD|\$|€|£|฿)\s?"
_MULT = r"(?:\s?(?:million|billion|thousand|ล้าน))?"

QUANTITY = re.compile(
    rf"(?P<cur>{_CURR})(?P<n1>{_NUM}){_MULT}"  # $250 billion
    rf"|(?:ร้อยละ\s?)(?P<n2>{_NUM})"  # ร้อยละ 25
    rf"|(?P<n3>{_NUM})(?P<unit>{_UNIT})(?![A-Za-z])",  # 4.5 %, 30 days
)
DATE = re.compile(
    r"\b(?P<d>[0-3]?\d)\s(?P<m>January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s(?P<y>(?:19|20)\d\d)\b"
)
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


@dataclass(frozen=True)
class Change:
    start: int
    end: int
    original: str
    perturbed: str


def _parse(num: str) -> tuple[float, int, str, str]:
    """-> (value, decimals, thousands_sep, decimal_sep) for 1,000.5 / 1 000,5 / 4,5 / 4.5."""
    raw = num.translate(_TO_ASCII)
    m = re.fullmatch(r"(\d{1,3}(?:([,\u00a0 ])\d{3})+)(?:([.,])(\d+))?", raw)
    if m:
        tsep, dsep = m.group(2), m.group(3) or ("," if m.group(2) != "," else ".")
        intpart, frac = m.group(1).replace(tsep, ""), m.group(4) or ""
    elif re.fullmatch(r"\d+,\d{1,2}", raw):
        tsep, dsep = "", ","
        intpart, frac = raw.split(",")
    else:
        tsep, dsep = "", "."
        intpart, _, frac = raw.partition(".")
    return float(f"{intpart}.{frac or 0}"), len(frac), tsep, dsep


def _format(val: float, decimals: int, tsep: str, dsep: str) -> str:
    out = f"{val:,.{decimals}f}"  # 1,234.50
    out = out.replace(",", "\0").replace(".", dsep).replace("\0", tsep)
    return out


def _scale(num: str, rng: random.Random) -> str:
    thai = any(c in _THAI for c in num)
    val, decimals, tsep, dsep = _parse(num)
    new = val
    for _ in range(20):
        f = rng.uniform(0.55, 0.85) if rng.random() < 0.5 else rng.uniform(1.2, 1.8)
        new = round(val * f, decimals) if val else float(rng.randint(1, 9))
        if decimals == 0:
            new = float(max(1, round(new)))
        if new != val:
            break
    out = _format(new, decimals, tsep, dsep)
    return out.translate(_TO_THAI) if thai else out


def perturb_numbers(text: str, seed: int = 0) -> tuple[str, list[Change]]:
    """Return ``(perturbed_text, changes)``; offsets in ``changes`` refer to the original text."""
    rng = random.Random(f"{seed}:{text}")
    edits: list[Change] = []
    for m in DATE.finditer(text):
        d, mo, y = int(m["d"]), _MONTHS.index(m["m"]), int(m["y"])
        nd = min(28, max(1, d + rng.choice([-1, 1]) * rng.randint(3, 11)))
        ny = y + rng.choice([-1, 1])
        edits.append(Change(m.start(), m.end(), m.group(), f"{nd} {_MONTHS[mo]} {ny}"))
    taken = [(e.start, e.end) for e in edits]
    for m in QUANTITY.finditer(text):
        grp = next(g for g in ("n1", "n2", "n3") if m[g] is not None)
        s, e = m.span(grp)
        if any(a <= s < b for a, b in taken):
            continue
        edits.append(Change(s, e, m[grp], _scale(m[grp], rng)))
    edits.sort(key=lambda c: c.start)
    out, pos = [], 0
    for c in edits:
        out.append(text[pos : c.start])
        out.append(c.perturbed)
        pos = c.end
    out.append(text[pos:])
    return "".join(out), edits
