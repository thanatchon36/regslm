"""Throughput / latency / memory aggregation for the RQ4 benchmark (docs/specs/quant_bench.md).

Pure functions (no GPU, no network) so they are unit-testable on CPU.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Gen:
    """One generation: text plus accounting."""

    text: str
    n_prompt: int
    n_gen: int
    latency_s: float


def pct(xs: Sequence[float], q: float) -> float | None:
    return float(np.quantile(list(xs), q)) if len(xs) else None


def timing_summary(gens: Sequence[Gen], wall_s: float) -> dict[str, Any]:
    """Offline throughput (wall-clock over the whole batch) + per-unit latency percentiles."""
    lat = [g.latency_s for g in gens]
    n_gen = sum(g.n_gen for g in gens)
    return {
        "n": len(gens),
        "wall_s": wall_s,
        "units_per_s": len(gens) / wall_s if wall_s > 0 else None,
        "gen_tokens": n_gen,
        "prompt_tokens": sum(g.n_prompt for g in gens),
        "gen_tok_per_s": n_gen / wall_s if wall_s > 0 else None,
        "latency_s": {
            "mean": float(np.mean(lat)) if lat else None,
            "p50": pct(lat, 0.5),
            "p95": pct(lat, 0.95),
        },
    }


def parse_gpu_csv(path: Path) -> list[tuple[str, float]]:
    """``nvidia-smi --query-gpu=timestamp,memory.used --format=csv`` trace (1 Hz) -> rows."""
    rows: list[tuple[str, float]] = []
    if not path.exists():
        return rows
    for r in csv.reader(path.read_text().splitlines()):
        if len(r) < 2:
            continue
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*MiB", r[1])
        if m:
            rows.append((r[0], float(m.group(1))))
    return rows


def vram_peak(path: Path, baseline_samples: int = 3) -> dict[str, float | None]:
    """Peak VRAM (MiB) from a gpu.csv trace; baseline = min of the first samples (D-014)."""
    rows = parse_gpu_csv(path)
    if not rows:
        return {"peak_mib": None, "baseline_mib": None, "peak_minus_baseline_mib": None}
    used = [v for _, v in rows]
    base = min(used[:baseline_samples])
    return {
        "peak_mib": max(used),
        "baseline_mib": base,
        "peak_minus_baseline_mib": max(used) - base,
    }


def read_vmhwm_kib(pid: int, proc: Path = Path("/proc")) -> int | None:
    """Peak resident set (VmHWM, KiB) of a process from /proc/<pid>/status."""
    try:
        text = (proc / str(pid) / "status").read_text()
    except OSError:
        return None
    m = re.search(r"^VmHWM:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def count_mismatches(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Greedy-output identity check (speculative vs target alone): by unit_id on ``completion``."""
    bm = {r["unit_id"]: r["completion"] for r in b}
    shared = [r for r in a if r["unit_id"] in bm]
    bad = [r["unit_id"] for r in shared if r["completion"] != bm[r["unit_id"]]]
    return {
        "n_compared": len(shared),
        "n_mismatch": len(bad),
        "mismatch_unit_ids": bad[:50],
        "n_only_in_one": len(a) + len(b) - 2 * len(shared),
    }


def write_result(out_dir: Path, backend: str, fmt: str, result: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{backend}-{fmt}.json"
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return p


def collect(bench_dir: Path) -> list[dict[str, Any]]:
    """All ``<backend>-<format>.json`` results under a bench dir, flattened to table rows."""
    out: list[dict[str, Any]] = []
    for p in sorted(bench_dir.glob("*.json")):
        r = json.loads(p.read_text())
        thr, single = r.get("throughput") or {}, r.get("single_stream") or {}
        ev = (r.get("eval") or {}).get("metrics", {})
        out.append(
            {
                "backend": r.get("backend"),
                "format": r.get("format"),
                "threads": r.get("threads"),
                "of1": ev.get("of1"),
                "validity": ev.get("validity_rate"),
                "units_per_s": thr.get("units_per_s"),
                "gen_tok_per_s": thr.get("gen_tok_per_s"),
                "p50_s": (single.get("latency_s") or {}).get("p50"),
                "p95_s": (single.get("latency_s") or {}).get("p95"),
                "vram_mib": (r.get("memory") or {}).get("vram_peak_minus_baseline_mib"),
                "host_ram_mib": (r.get("memory") or {}).get("host_ram_peak_mib"),
                "size_gb": r.get("model_size_gb"),
            }
        )
    return out


def to_markdown(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])

    def f(v: Any) -> str:
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
