"""Statistics for evaluation (docs/RESEARCH_QUESTIONS.md section 0 "Statistics").

scipy is not a dependency. Student-t quantiles use an exact table for the two-sided 95 %
level (df 1..30, 40, 60, 120) and, for any other level, the Cornish-Fisher expansion
(Abramowitz & Stegun 26.7.5) around ``statistics.NormalDist``; the latter is accurate to
about 1e-3 for df >= 5 and is only a fallback.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

_T975 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]  # fmt: skip
_T975_SPARSE = [(40, 2.021), (60, 2.000), (120, 1.980)]


def t_quantile(p: float, df: int) -> float:
    """Student-t quantile at probability ``p`` (see module docstring for accuracy)."""
    if not 0.5 < p < 1.0:
        raise ValueError("p must be in (0.5, 1)")
    if df < 1:
        raise ValueError("df must be >= 1")
    if abs(p - 0.975) < 1e-12:
        if df <= 30:
            return _T975[df - 1]
        for d, v in _T975_SPARSE:
            if df <= d:
                return v
        return 1.960
    z = statistics.NormalDist().inv_cdf(p)
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    return z + g1 / df + g2 / df**2 + g3 / df**3


MetricFn = Callable[[pl.DataFrame], float]


def _resample_indices(n: int, reps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def bootstrap_ci(
    per_unit_df: pl.DataFrame,
    metric_fn: MetricFn,
    n: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile unit-bootstrap CI; returns (point estimate, lo, hi)."""
    point = metric_fn(per_unit_df)
    if per_unit_df.height == 0:
        return point, math.nan, math.nan
    idx = _resample_indices(per_unit_df.height, n, seed)
    vals = np.array([metric_fn(per_unit_df[row.tolist()]) for row in idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


class PairedResult(NamedTuple):
    diff: float
    lo: float
    hi: float
    p: float


def paired_bootstrap_diff(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    metric_fn: MetricFn,
    n: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedResult:
    """Paired bootstrap of ``metric(A) - metric(B)`` with shared resampled unit indices.

    Units are aligned on ``unit_id``. ``p`` is two-sided: 2 * min(P(d* <= 0), P(d* >= 0)),
    capped at 1.
    """
    a = df_a.sort("unit_id")
    b = df_b.sort("unit_id")
    if a["unit_id"].to_list() != b["unit_id"].to_list():
        raise ValueError("systems must be evaluated on the same unit_ids")
    diff = metric_fn(a) - metric_fn(b)
    if a.height == 0:
        return PairedResult(diff, math.nan, math.nan, math.nan)
    idx = _resample_indices(a.height, n, seed)
    d = np.empty(n)
    for k, row in enumerate(idx):
        r = row.tolist()
        d[k] = metric_fn(a[r]) - metric_fn(b[r])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
    return PairedResult(diff, float(lo), float(hi), p)


def seed_mean_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and t-interval over per-seed values; (mean, nan, nan) for a single seed."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    if n == 1:
        return mean, math.nan, math.nan
    half = t_quantile(1 - alpha / 2, n - 1) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1), input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def crossover(
    n_values: Sequence[int], diffs_with_ci: Sequence[tuple[float, float, float]]
) -> int | None:
    """RQ1 rule: first N whose CI excludes 0 with the student better (lo > 0) at N and at the
    next larger N. ``diffs_with_ci`` holds (diff, lo, hi) per N. None if never (the largest N
    has no successor, so it cannot start a crossover)."""
    pts = sorted(zip(n_values, diffs_with_ci, strict=True), key=lambda t: t[0])
    for k in range(len(pts) - 1):
        if pts[k][1][1] > 0 and pts[k + 1][1][1] > 0:
            return pts[k][0]
    return None


HEADLINE = ("of1", "precision", "recall", "exact_obligation_rate", "numeric_exact", "validity_rate")


def sanity_flags(
    metrics: Mapping[str, float],
    parent: Mapping[str, float] | None = None,
    ci_width: float = 0.0,
) -> list[str]:
    """Anomalies per MISSION section 0: any metric exactly 0, 1 or 100 (rates such as
    ``validity_rate`` = 1.0 included — they must be explained in docs/ANOMALIES.md), any
    ``field_f1.*`` above 0.98, and (when ``parent`` is the bf16 parent's metrics) a quantized
    ``of1`` or ``numeric_exact`` exceeding the parent's by more than ``ci_width``."""
    flags: list[str] = []
    for k, v in metrics.items():
        if v in (0.0, 1.0, 100.0):
            flags.append(f"{k}={v:g} exactly")
        if k.startswith("field_f1.") and 0.98 < v < 1.0:
            flags.append(f"{k}={v:.3f} > 0.98")
    if parent is not None:
        for k in ("of1", "numeric_exact"):
            if k in metrics and k in parent and metrics[k] - parent[k] > ci_width:
                flags.append(
                    f"{k} beats parent by {metrics[k] - parent[k]:.4f} > CI width {ci_width:.4f}"
                )
    return flags
