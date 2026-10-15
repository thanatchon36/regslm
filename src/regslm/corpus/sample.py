"""Stratified sampling of units for the gold set and the trigger filter (paper §3.3, D-022).

Gold allocation (300 units):
- by language: Thai gets ``thai_share`` of the sample if Thai units exist (default 20 %),
  so RQ5 has gold evidence; the rest is English;
- within a language, by document, proportional to sqrt(#units) — large instruments such as
  12 CFR 1026 (Reg Z) would otherwise dominate;
- within a document, ``untriggered_share`` (15 %) of the draw comes from units with no
  deontic trigger in text + heading, so the trigger pre-filter's miss rate can be measured
  on gold.

Deterministic given ``seed``. Gold units are written with their ids to
``data/gold/sample.jsonl`` and are excluded from every training split.
"""

from __future__ import annotations

import math
import re

import polars as pl

TRIGGER = re.compile(
    r"\b(shall|must|may|should|required?|requires|prohibit\w*|expected|obliged|permitted|"
    r"entitled|ensure|need not|is to|are to)\b"
    r"|ต้อง|ห้าม|มิให้|อาจ|มีสิทธิ|ควร|ให้",
    re.IGNORECASE,
)


def has_trigger(text: str, heading: str = "") -> bool:
    """True if text or its lead-in heading contains a deontic trigger (en or th)."""
    return TRIGGER.search(f"{text} {heading}") is not None


def add_trigger_column(units: pl.DataFrame) -> pl.DataFrame:
    pairs = zip(units["text"], units["heading"], strict=True)
    flags = [has_trigger(t or "", h or "") for t, h in pairs]
    return units.with_columns(pl.Series("trigger", flags, dtype=pl.Boolean))


def _allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation proportional to sqrt(size), at least 1 per stratum."""
    weights = {k: math.sqrt(v) for k, v in sizes.items() if v > 0}
    wsum = sum(weights.values())
    raw = {k: total * w / wsum for k, w in weights.items()}
    alloc = {k: max(1, math.floor(r)) for k, r in raw.items()}
    rest = total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - math.floor(raw[k]), reverse=True)[: max(0, rest)]:
        alloc[k] += 1
    return {k: min(n, sizes[k]) for k, n in alloc.items()}


def stratified_gold(
    units: pl.DataFrame,
    n: int = 300,
    thai_share: float = 0.2,
    untriggered_share: float = 0.15,
    seed: int = 0,
) -> pl.DataFrame:
    """Sample ``n`` units (fewer if the corpus is small), adding a ``stratum`` column."""
    if "trigger" not in units.columns:
        units = add_trigger_column(units)
    n_th = round(n * thai_share) if (units["language"] == "th").any() else 0
    by_lang = {"th": n_th, "en": n - n_th}
    parts: list[pl.DataFrame] = []
    for lang, n_lang in by_lang.items():
        sub = units.filter(pl.col("language") == lang)
        if n_lang == 0 or sub.height == 0:
            continue
        sizes = dict(sub.group_by("doc_id").len().iter_rows())
        for doc, n_doc in sorted(_allocate(sizes, n_lang).items()):
            d = sub.filter(pl.col("doc_id") == doc)
            n_un = round(n_doc * untriggered_share)
            for trig, k in ((False, n_un), (True, n_doc - n_un)):
                pool = d.filter(pl.col("trigger") == trig)
                k = min(k, pool.height)
                if k:
                    parts.append(
                        pool.sample(k, seed=seed, shuffle=True).with_columns(
                            pl.lit(f"{lang}/{doc}/{'trig' if trig else 'notrig'}").alias("stratum")
                        )
                    )
    return pl.concat(parts).sort("unit_id") if parts else units.head(0)
