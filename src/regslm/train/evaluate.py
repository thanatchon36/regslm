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


def evaluate_rows(rows: list[dict[str, Any]], n_boot: int = 10_000) -> dict[str, Any]:
    results: list[M.UnitResult] = []
    mem: list[M.MemRow] = []
    n_missing_ref = 0
    for r in rows:
        if not r.get("reference"):
            n_missing_ref += 1
            continue
        ref = Extraction.model_validate(r["reference"])
        pred = Extraction.model_validate(r["extraction"]) if r.get("valid") else None
        results.append(M.UnitResult(r["unit_id"], ref, pred, int(r.get("n_unresolved", 0))))
        if r.get("memorisation_targets"):
            mem.append(M.MemRow(ref, pred, r["memorisation_targets"]))
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
    for f in files:
        rows = [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        res["sets"][f.stem] = evaluate_rows(rows, n_boot)
    (run_dir / "eval.json").write_text(json.dumps(res, indent=2))
    return res
