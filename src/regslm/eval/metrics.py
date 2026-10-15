"""Obligation- and field-level metrics (docs/RESEARCH_QUESTIONS.md section 0).

Conventions chosen where docs/specs/eval.md is silent:

- ``exact_obligation_rate`` = matched pairs with every field equal / reference obligations.
- ``numeric_exact`` = among matched pairs whose gold has a numeric component (a threshold, or a
  deadline containing a (number, time-unit) pair), the fraction where every such component is
  equal (threshold exact, deadline relaxed).
- ``empty_rate`` = fraction of units whose prediction is valid but has no obligations at all.
- ``hallucination_rate`` = FP / predicted (predicted = matched + unmatched + unresolved).
- Field F1 compares ``deadline`` strictly (normalised string); numeric-exact uses the relaxed form.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import polars as pl

from regslm.rewards.normalize import (
    deadline_pairs,
    deadlines_equal,
    deadlines_equal_relaxed,
    match_obligations,
    normalize_text,
    thresholds_equal,
)
from regslm.schema.obligation import Extraction, Obligation

FIELDS = ("subject", "action", "object", "condition", "threshold", "deadline", "exceptions")


class UnitResult(NamedTuple):
    unit_id: str
    reference: Extraction
    prediction: Extraction | None  # None = invalid output
    n_unresolved: int = 0  # generated obligations whose quote did not resolve (D-020)


class PRF(NamedTuple):
    precision: float
    recall: float
    f1: float


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def prf(tp: float, fp: float, fn: float) -> PRF:
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return PRF(p, r, _div(2 * p * r, p + r))


# ---------------------------------------------------------------------------- field helpers


def _nonempty(o: Obligation, f: str) -> bool:
    v = getattr(o, f)
    return v is not None and bool(v)


def _field_equal(a: Obligation, b: Obligation, f: str) -> bool:
    if f == "threshold":
        return thresholds_equal(a.threshold, b.threshold)
    if f == "exceptions":
        return {normalize_text(x) for x in a.exceptions} == {
            normalize_text(x) for x in b.exceptions
        }
    if f == "deadline":
        return deadlines_equal(a.deadline, b.deadline)
    return normalize_text(str(getattr(a, f))) == normalize_text(str(getattr(b, f)))


def _numeric_status(gold: Obligation, pred: Obligation) -> tuple[bool, bool]:
    """(gold has a numeric component, all numeric components equal)."""
    parts: list[bool] = []
    if gold.threshold is not None:
        parts.append(thresholds_equal(gold.threshold, pred.threshold))
    if deadline_pairs(gold.deadline):
        parts.append(deadlines_equal_relaxed(gold.deadline, pred.deadline))
    return bool(parts), all(parts)


# ---------------------------------------------------------------------------- per-unit table


def _unit_row(u: UnitResult) -> dict[str, Any]:
    ref = u.reference.obligations
    pred = u.prediction.obligations if u.prediction is not None else []
    pairs = match_obligations(pred, ref)
    mp = {i for i, _ in pairs}
    mg = {j for _, j in pairs}
    row: dict[str, Any] = {
        "unit_id": u.unit_id,
        "n_ref": len(ref),
        "n_pred": len(pred) + u.n_unresolved,
        "tp": len(pairs),
        "fp": len(pred) - len(pairs) + u.n_unresolved,
        "fn": len(ref) - len(pairs),
        "valid": int(u.prediction is not None),
        "empty": int(u.prediction is not None and not pred and u.n_unresolved == 0),
        "exact": 0,
        "num_n": 0,
        "num_ok": 0,
    }
    for f in FIELDS:
        tp = fp = fn = 0
        for i, j in pairs:
            p, g = pred[i], ref[j]
            pn, gn = _nonempty(p, f), _nonempty(g, f)
            eq = pn and gn and _field_equal(p, g, f)
            tp += eq
            fp += pn and not eq
            fn += gn and not eq
        fp += sum(_nonempty(o, f) for i, o in enumerate(pred) if i not in mp)
        fn += sum(_nonempty(o, f) for j, o in enumerate(ref) if j not in mg)
        row[f"{f}_tp"], row[f"{f}_fp"], row[f"{f}_fn"] = tp, fp, fn
    for i, j in pairs:
        if all(_field_equal(pred[i], ref[j], f) for f in FIELDS):
            row["exact"] += 1
        numeric, ok = _numeric_status(ref[j], pred[i])
        row["num_n"] += numeric
        row["num_ok"] += numeric and ok
    return row


def per_unit_counts(results: Iterable[UnitResult]) -> pl.DataFrame:
    """One row per unit with TP/FP/FN (obligation level and per field) and auxiliary counts."""
    rows = [_unit_row(u) for u in results]
    if not rows:  # keep the schema so downstream code can run on an empty set
        rows = [_unit_row(UnitResult("", Extraction(unit_id=""), None))]
        return pl.DataFrame(rows).clear()
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------- aggregates


def _sum(df: pl.DataFrame, col: str) -> int:
    return int(df[col].sum()) if df.height else 0


def prf_from_counts(df: pl.DataFrame) -> PRF:
    """Micro P/R/F1 over a per-unit count table."""
    return prf(_sum(df, "tp"), _sum(df, "fp"), _sum(df, "fn"))


def field_f1_from_counts(df: pl.DataFrame) -> dict[str, PRF]:
    return {f: prf(_sum(df, f"{f}_tp"), _sum(df, f"{f}_fp"), _sum(df, f"{f}_fn")) for f in FIELDS}


def obligation_prf(results: Sequence[UnitResult]) -> PRF:
    """Micro-averaged obligation precision, recall and F1 (OF1)."""
    return prf_from_counts(per_unit_counts(results))


def field_f1(results: Sequence[UnitResult]) -> dict[str, PRF]:
    """Per-field P/R/F1 over matched pairs plus unmatched obligations' non-empty fields."""
    return field_f1_from_counts(per_unit_counts(results))


def summarize(df: pl.DataFrame) -> dict[str, float]:
    """Headline metrics from a per-unit table (keys: of1, precision, recall, ..., field_f1.<f>)."""
    o = prf_from_counts(df)
    n_units = df.height
    out = {
        "of1": o.f1,
        "precision": o.precision,
        "recall": o.recall,
        "exact_obligation_rate": _div(_sum(df, "exact"), _sum(df, "n_ref")),
        "numeric_exact": _div(_sum(df, "num_ok"), _sum(df, "num_n")),
        "validity_rate": _div(_sum(df, "valid"), n_units),
        "hallucination_rate": _div(_sum(df, "fp"), _sum(df, "n_pred")),
        "empty_rate": _div(_sum(df, "empty"), n_units),
    }
    for f, s in field_f1_from_counts(df).items():
        out[f"field_f1.{f}"] = s.f1
    return out


def metric(name: str) -> Callable[[pl.DataFrame], float]:
    """A ``metric_fn`` for ``bootstrap_ci`` selecting ``summarize(df)[name]``."""
    return lambda df: summarize(df)[name]


# ---------------------------------------------------------------------------- memorisation


@dataclass(frozen=True)
class MemRow:
    """A ``test-perturbed`` unit: perturbed reference, model prediction, memorisation targets
    as stored by ``data/perturb_set.py`` (``[obligation_idx, field, original, perturbed]``)."""

    reference: Extraction
    prediction: Extraction | None
    targets: Sequence[Sequence[Any]]


def memorisation_rate(rows: Iterable[MemRow]) -> dict[str, float]:
    """Fraction of targets whose matched prediction carries the original (memorised) value,
    and the fraction carrying the perturbed (correct) value. Unmatched targets count in the
    denominator but in neither numerator. Only ``threshold.value`` targets are scored."""
    n = mem = ok = 0
    for r in rows:
        pred = r.prediction.obligations if r.prediction is not None else []
        by_gold = {j: i for i, j in match_obligations(pred, r.reference.obligations)}
        for idx, fld, orig, pert in r.targets:
            if fld != "threshold.value":
                continue
            n += 1
            pi = by_gold.get(int(idx))
            th = pred[pi].threshold if pi is not None else None
            if th is None:
                continue
            mem += math.isclose(th.value, float(orig), rel_tol=1e-9, abs_tol=1e-12)
            ok += math.isclose(th.value, float(pert), rel_tol=1e-9, abs_tol=1e-12)
    return {
        "memorisation_rate": _div(mem, n),
        "perturbed_correct_rate": _div(ok, n),
        "n_targets": float(n),
    }
