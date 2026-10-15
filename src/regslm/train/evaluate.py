"""Score a run's predictions with ``regslm.eval`` and write ``eval.json`` (docs/specs/train.md)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from regslm.eval import metrics as M
from regslm.eval import stats as S
from regslm.schema.obligation import Extraction
from regslm.train.config import RUNS_DIR


def _pct(xs: list[float], q: float) -> float:
    return float(np.quantile(xs, q)) if xs else float("nan")


def bootstrap_all(
    df: Any, n: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict[str, tuple[float, float, float]]:
    """Unit-bootstrap CI for every ``summarize`` key in one pass.

    Uses the same resample indices as ``eval.stats.bootstrap_ci`` (same rng call), so the
    result equals per-metric ``bootstrap_ci`` calls at a fraction of the cost.
    """
    point = M.summarize(df)
    if df.height == 0 or n <= 0:
        return {k: (v, float("nan"), float("nan")) for k, v in point.items()}
    idx = np.random.default_rng(seed).integers(0, df.height, size=(n, df.height))
    draws = [M.summarize(df[row.tolist()]) for row in idx]
    out = {}
    for k, v in point.items():
        lo, hi = np.quantile([d[k] for d in draws], [alpha / 2, 1 - alpha / 2])
        out[k] = (v, float(lo), float(hi))
    return out


def _parse_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[M.UnitResult], list[tuple[str, M.MemRow]], int]:
    results: list[M.UnitResult] = []
    mem: list[tuple[str, M.MemRow]] = []
    n_missing_ref = 0
    for r in rows:
        if not r.get("reference"):
            n_missing_ref += 1
            continue
        ref = Extraction.model_validate(r["reference"])
        pred = Extraction.model_validate(r["extraction"]) if r.get("valid") else None
        results.append(M.UnitResult(r["unit_id"], ref, pred, int(r.get("n_unresolved", 0))))
        if r.get("memorisation_targets"):
            mem.append((r["unit_id"], M.MemRow(ref, pred, r["memorisation_targets"])))
    return results, mem, n_missing_ref


def decomposition_ci(
    unit_rows: list[tuple[str, M.MemRow]], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict[str, Any]:
    """4-way memorisation decomposition with a bootstrap CI over *units* (rows sharing a
    ``unit_id``, e.g. the two perturbation seeds, are resampled together)."""
    per_unit: dict[str, np.ndarray] = {}
    for uid, r in unit_rows:
        v = per_unit.setdefault(uid, np.zeros(len(M.MEM_CATEGORIES)))
        for c in M.memorisation_outcomes(r):
            v[M.MEM_CATEGORIES.index(c)] += 1
    mat = np.array([v for v in per_unit.values() if v.sum() > 0])
    out: dict[str, Any] = {"n_units": len(mat), "n_targets": int(mat.sum()) if len(mat) else 0}
    if not len(mat):
        for c in M.MEM_CATEGORIES:
            out[c] = {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return out
    point = mat.sum(0) / mat.sum()
    idx = np.random.default_rng(seed).integers(0, len(mat), size=(n_boot, len(mat)))
    sums = mat[idx].sum(1)  # (n_boot, 4)
    frac = sums / np.maximum(sums.sum(1, keepdims=True), 1)
    lo, hi = np.quantile(frac, [alpha / 2, 1 - alpha / 2], axis=0)
    for k, c in enumerate(M.MEM_CATEGORIES):
        out[c] = {"point": float(point[k]), "lo": float(lo[k]), "hi": float(hi[k])}
    return out


def paired_of1(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], n_boot: int = 10_000
) -> dict[str, Any]:
    """Paired (same unit_ids) OF1 difference A - B with a paired unit bootstrap."""
    ra, _, _ = _parse_rows(rows_a)
    rb, _, _ = _parse_rows(rows_b)
    common = {r.unit_id for r in ra} & {r.unit_id for r in rb}
    da = M.per_unit_counts([r for r in ra if r.unit_id in common])
    db = M.per_unit_counts([r for r in rb if r.unit_id in common])
    p = S.paired_bootstrap_diff(da, db, M.metric("of1"), n=n_boot)
    return {
        "n_units": len(common), "of1_a": M.summarize(da)["of1"], "of1_b": M.summarize(db)["of1"],
        "diff": p.diff, "lo": p.lo, "hi": p.hi, "p": p.p, "n_boot": n_boot,
    }  # fmt: skip


def evaluate_rows(rows: list[dict[str, Any]], n_boot: int = 10_000) -> dict[str, Any]:
    results, unit_mem, n_missing_ref = _parse_rows(rows)
    mem = [m for _, m in unit_mem]
    df = M.per_unit_counts(results)
    cis = bootstrap_all(df, n_boot)
    summary = M.summarize(df)
    lat = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]
    out: dict[str, Any] = {
        "n_units": len(results),
        "n_skipped_no_reference": n_missing_ref,
        "metrics": summary,
        "ci95": {k: {"point": v[0], "lo": v[1], "hi": v[2]} for k, v in cis.items()},
        "n_boot": n_boot,
        "n_unresolved_total": int(sum(int(r.get("n_unresolved", 0)) for r in rows)),
        "latency_s": {
            "mean": statistics.fmean(lat) if lat else None,
            "p50": _pct(lat, 0.5),
            "p95": _pct(lat, 0.95),
        },
        "gen_tokens_mean": statistics.fmean([r["n_gen_tokens"] for r in rows]) if rows else None,
        "sanity_flags": S.sanity_flags(summary),
    }
    if mem:
        out["memorisation"] = M.memorisation_rate(mem)
        out["memorisation_decomposition"] = decomposition_ci(unit_mem, n_boot)
    return out


def evaluate(
    run_id: str, sets: list[str] | None = None, n_boot: int = 10_000, runs: Path = RUNS_DIR
) -> dict[str, Any]:
    run_dir = runs / run_id
    files = sorted((run_dir / "predictions").glob("*.jsonl"))
    if sets:
        files = [f for f in files if f.stem in sets]
    if not files:
        raise FileNotFoundError(f"no predictions under {run_dir / 'predictions'}")
    res: dict[str, Any] = {"run_id": run_id, "sets": {}}
    raw: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        raw[f.stem] = rows
        res["sets"][f.stem] = evaluate_rows(rows, n_boot)
    # Contamination analyses (docs/RESEARCH_QUESTIONS.md section 0, headline rule).
    if "num440" in raw and "num440_perturbed" in raw:
        res["num440_paired"] = paired_of1(raw["num440"], raw["num440_perturbed"], n_boot)
    pool = [n for n in ("mr_pool_s0", "mr_pool_s1") if n in raw]
    if pool:
        pooled = [m for n in pool for m in _parse_rows(raw[n])[1]]
        res["mr_pool_decomposition"] = {
            "sets": pool,
            **decomposition_ci(pooled, n_boot),
        }
    (run_dir / "eval.json").write_text(json.dumps(res, indent=2))
    return res
