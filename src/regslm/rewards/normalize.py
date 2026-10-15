"""Scoring normalisation and obligation matching (docs/ANNOTATION_GUIDE.md section 4).

Public helpers, shared by ``regslm.rewards`` and ``regslm.eval``:

- :func:`normalize_text`: NFKC, lowercase, whitespace collapse, trailing punctuation and
  (optionally) leading-article stripping.
- :func:`thresholds_equal`: value (1e-9 relative), aliased unit, and comparator equality.
- :func:`deadline_pairs` / :func:`deadlines_equal_relaxed`: relaxed (number, time-unit) metric.
- :func:`extract_numbers`: numbers occurring in a text (Thai digits, thousands separators).
- :func:`match_obligations`: exact maximum-weight assignment (Hungarian algorithm,
  implemented here; scipy is not a dependency) on span IoU >= 0.5 with equal modality.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache

from regslm.schema.obligation import Obligation, Span, Threshold

IOU_THRESHOLD = 0.5
REL_TOL = 1e-9

_ARTICLES = ("a ", "an ", "the ")
_TRAILING_PUNCT = ".,;:!?)]}\"'”’、。"
_UNIT_ALIASES = {
    "%": "%",
    "per cent": "%",
    "percent": "%",
    "percentage": "%",
    "ร้อยละ": "%",
    "เปอร์เซ็นต์": "%",
}
_TIME_UNITS = {
    "day": "day",
    "days": "day",
    "business day": "bday",
    "business days": "bday",
    "working day": "bday",
    "working days": "bday",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "hour": "hour",
    "hours": "hour",
    "วัน": "day",
    "สัปดาห์": "week",
    "เดือน": "month",
    "ปี": "year",
    "ชั่วโมง": "hour",
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "sixty": 60, "ninety": 90,
}  # fmt: skip
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_UNIT_PATTERN = "|".join(sorted((re.escape(u) for u in _TIME_UNITS), key=len, reverse=True))
_DEADLINE_RE = re.compile(
    rf"(\d+(?:\.\d+)?|{'|'.join(_NUMBER_WORDS)})\s*-?\s*({_UNIT_PATTERN})(?![a-z])"
)


def normalize_text(s: str, *, strip_articles: bool = False) -> str:
    """Guide section 4 string normalisation."""
    t = unicodedata.normalize("NFKC", s).lower()
    t = " ".join(t.split())
    t = t.rstrip(_TRAILING_PUNCT).rstrip()
    if strip_articles:
        for art in _ARTICLES:
            if t.startswith(art):
                t = t[len(art) :].lstrip()
                break
    return t


def normalize_unit(unit: str) -> str:
    """Normalise a threshold unit through the alias map (% = per cent = percent = ร้อยละ)."""
    t = normalize_text(unit)
    return _UNIT_ALIASES.get(t, t)


_TH_NUM_WORDS = frozenset(
    [
        "ศูนย์",
        "หนึ่ง",
        "เอ็ด",
        "สอง",
        "ยี่",
        "สาม",
        "สี่",
        "ห้า",
        "หก",
        "เจ็ด",
        "แปด",
        "เก้า",
        "สิบ",
        "ร้อย",
        "พัน",
        "หมื่น",
        "แสน",
        "ล้าน",
    ]
)  # fmt: skip
_TH_NUM_CHARS = re.compile(r"^[ก-๛]+$")


def _is_th_number_word(tok: str) -> bool:
    """True if ``tok`` is fully made of Thai numeral words (greedy longest match)."""
    i = 0
    while i < len(tok):
        for w in sorted(_TH_NUM_WORDS, key=len, reverse=True):
            if tok.startswith(w, i):
                i += len(w)
                break
        else:
            return False
    return bool(tok)


@lru_cache(maxsize=8192)
def thai_number_words_to_digits(s: str) -> str:
    """Replace runs of Thai number words ("สามสิบ", "ยี่สิบห้า") by Arabic digits.

    Works on whole tokens from PyThaiNLP word segmentation, so words that merely contain a
    numeral syllable ("สามารถ", "ร้อยละ") are untouched. Returns ``s`` unchanged if PyThaiNLP is
    not installed or the text has no Thai.
    """
    if not re.search(r"[ก-๛]", s):
        return s
    try:
        from pythainlp.tokenize import word_tokenize
        from pythainlp.util import thaiword_to_num
    except ImportError:  # pragma: no cover - th extra not installed
        return s
    toks = word_tokenize(s, engine="newmm", keep_whitespace=True)
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            try:
                out.append(str(thaiword_to_num("".join(run))))
            except (ValueError, KeyError, TypeError):
                out.extend(run)
            run.clear()

    for t in toks:
        if t != "ร้อยละ" and _is_th_number_word(t):
            run.append(t)
        else:
            flush()
            out.append(t)
    flush()
    return "".join(out)


def digits_to_ascii(s: str) -> str:
    """NFKC, Thai digits to ASCII, thousands separators removed, EU decimal comma to point."""
    t = thai_number_words_to_digits(unicodedata.normalize("NFKC", s).translate(_THAI_DIGITS))
    t = re.sub(r"(?<=\d)[,\u00a0 ](?=\d{3}(?!\d))", "", t)  # 1,000 / 5 000 000
    return re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", t)  # EU decimal comma: 4,5 -> 4.5


def extract_numbers(text: str) -> list[float]:
    """All numbers occurring in ``text`` after digit and thousands-separator normalisation."""
    return [float(m) for m in _NUM_RE.findall(digits_to_ascii(text))]


def number_in_text(value: float, text_numbers: Sequence[float]) -> bool:
    """True if ``value`` equals (1e-9 relative) one of ``text_numbers``."""
    return any(math.isclose(value, n, rel_tol=REL_TOL, abs_tol=1e-12) for n in text_numbers)


def thresholds_equal(a: Threshold | None, b: Threshold | None) -> bool:
    """Exact threshold match; both ``None`` counts as equal."""
    if a is None or b is None:
        return a is None and b is None
    return (
        math.isclose(a.value, b.value, rel_tol=REL_TOL, abs_tol=1e-12)
        and normalize_unit(a.unit) == normalize_unit(b.unit)
        and a.comparator == b.comparator
    )


def deadline_pairs(deadline: str) -> frozenset[tuple[float, str]]:
    """(number, canonical time-unit) pairs found in a deadline string."""
    t = digits_to_ascii(normalize_text(deadline))
    out: set[tuple[float, str]] = set()
    for num, unit in _DEADLINE_RE.findall(t):
        n = float(num) if num[0].isdigit() else float(_NUMBER_WORDS[num])
        out.add((n, _TIME_UNITS[unit]))
    return frozenset(out)


def deadlines_equal(a: str, b: str) -> bool:
    """Strict deadline match on normalised strings."""
    return normalize_text(a) == normalize_text(b)


def deadlines_equal_relaxed(a: str, b: str) -> bool:
    """Relaxed match: equal (number, time-unit) pairs; falls back to strict equality
    when neither string contains such a pair."""
    pa, pb = deadline_pairs(a), deadline_pairs(b)
    if not pa and not pb:
        return deadlines_equal(a, b)
    return pa == pb


def span_iou(a: Span, b: Span) -> float:
    """Intersection-over-union of two half-open character spans."""
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union else 0.0


def hungarian(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment on a rectangular matrix; returns (row, col) pairs.

    Exact O(n^2 m) Hungarian algorithm (potentials formulation); every row of the
    smaller dimension is assigned.
    """
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0:
        return []
    if n > m:
        flipped = hungarian([[cost[i][j] for i in range(n)] for j in range(m)])
        return [(i, j) for j, i in flipped]
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j])


def match_obligations(
    pred: Sequence[Obligation], gold: Sequence[Obligation]
) -> list[tuple[int, int]]:
    """Match predictions to gold obligations; returns sorted (pred_idx, gold_idx) pairs.

    Eligible pairs have equal modality and span IoU >= 0.5. Among eligible pairs the
    assignment maximises the number of matches, then total IoU. Unmatched predictions
    are false positives.
    """
    if not pred or not gold:
        return []
    big = 1000.0  # dominates any IoU sum, so cardinality is maximised first
    iou = [
        [
            span_iou(p.citation.span, g.citation.span) if p.modality == g.modality else 0.0
            for g in gold
        ]
        for p in pred
    ]
    cost = [[-(big + x) if x >= IOU_THRESHOLD else 0.0 for x in row] for row in iou]
    return [(i, j) for i, j in hungarian(cost) if iou[i][j] >= IOU_THRESHOLD]
