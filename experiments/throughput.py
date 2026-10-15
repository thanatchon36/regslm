"""Measured throughput -> ``results/throughput.json`` (feeds the re-projection in ``budget.py``).

This module reads *only* throughput fields: ``metrics.json:tokens_seen`` / ``train_wall_s`` and
``predict_meta.json:sets.*.{n,wall_s}``. It reads no quality metrics (no eval.json, no scores) and
is deliberately separate from the budget decision code.

Training tokens/s are stored per base as a LoRA-equivalent rate (measured x method slowdown) so
that ``grid.Rates.train_h``, which re-applies the slowdown, is not double-counted; the raw
per-method rates are kept under ``train_tok_s_by_method``. Runs with N >= 10k and seed 0 are used
when available (RESEARCH_QUESTIONS §7), else every finished run of that base.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from experiments import grid as G

RUNS_DIR = Path("results/runs")
OUT = Path("results/throughput.json")


def _key_of(g: dict[str, Any]) -> dict[str, str]:
    return {v["hf"]: k for k, v in g["bases"].items()} | {v: k for k, v in G.TEACHER_BASES.items()}


def measure(g: dict[str, Any], runs: Path = RUNS_DIR) -> dict[str, Any]:
    key_of = _key_of(g)
    train: dict[tuple[str, str], list[tuple[bool, float, float]]] = {}
    infer: dict[str, list[float]] = {}  # base -> [units, wall_s]
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        try:
            cfg = yaml.safe_load((d / "config.yaml").read_text())
        except OSError:
            continue
        base = key_of.get(str(cfg.get("base")))
        if base is None:
            continue
        mp = d / "metrics.json"
        if (d / "DONE").exists() and mp.exists() and not (d / "grpo.yaml").exists():
            m = json.loads(mp.read_text())
            tok, wall = m.get("tokens_seen"), m.get("train_wall_s")
            if tok and wall:
                pref = cfg.get("n_labels", 0) >= 10000 and cfg.get("seed", 0) == 0
                train.setdefault((base, str(cfg.get("method"))), []).append((pref, tok, wall))
        pm = d / "predict_meta.json"
        if pm.exists():
            for s in json.loads(pm.read_text()).get("sets", {}).values():
                if s.get("n") and s.get("wall_s"):
                    acc = infer.setdefault(base, [0.0, 0.0])
                    acc[0] += s["n"]
                    acc[1] += s["wall_s"]
    by_method: dict[str, dict[str, float]] = {}
    for (base, method), rows in train.items():
        use = [r for r in rows if r[0]] or rows
        by_method.setdefault(base, {})[method] = sum(r[1] for r in use) / sum(r[2] for r in use)
    tok_s: dict[str, float] = {}
    for base, mm in by_method.items():
        method = "lora" if "lora" in mm else sorted(mm)[0]
        tok_s[base] = mm[method] * G.SLOWDOWN.get(method, 1.0)
    return {
        "train_tok_s": tok_s,
        "train_tok_s_by_method": by_method,
        "infer_units_h": {b: u / w * 3600 for b, (u, w) in infer.items()},
    }


def write(
    g: dict[str, Any] | None = None, runs: Path = RUNS_DIR, out: Path = OUT
) -> dict[str, Any]:
    res = measure(g or G.load_grid(), runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
