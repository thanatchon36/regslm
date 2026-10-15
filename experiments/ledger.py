"""GPU-hour ledger: sum ``wall_s`` of every ``exit.json`` under ``results/runs/``.

Teacher-labelling runs (ids starting ``silver-``, ``gold-pass-``, ``teacher-``) are reported
separately (cap 12 h); ``quick`` smoke runs go to ``dev_h``. Everything else is an experiment and
is attributed to a block by its run-id prefix. Sub-run markers (``<id>/predict/exit.json``,
``<id>/evaluate/exit.json``) count towards ``<id>``. Reads only ``exit.json`` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEACHER_PREFIXES = ("silver-", "gold-pass-", "teacher-")
DEV_RUNS = ("quick",)
FIXED_PREFIX = {"zs": "zero_shot", "grpo": "grpo", "q": "quantization"}
RUNS_DIR = Path("results/runs")
LEDGER_PATH = Path("results/gpu_ledger.json")


def block_of(run_id: str, blocks: set[str] | None = None) -> str:
    """Block from the run-id prefix (``<block>-...``, ``zs-``, ``grpo-``, ``q-``)."""
    head = run_id.split("-", 1)[0]
    if head in FIXED_PREFIX:
        return FIXED_PREFIX[head]
    return head if blocks is None or head in blocks else "other"


def compute(
    runs: Path = RUNS_DIR, blocks: set[str] | None = None, teacher_cap_h: float = 12.0
) -> dict[str, Any]:
    teacher = dev = 0.0
    by_block: dict[str, float] = {}
    for p in sorted(runs.glob("*/exit.json")) + sorted(runs.glob("*/*/exit.json")):
        run_id = p.relative_to(runs).parts[0]
        try:
            wall = float(json.loads(p.read_text())["wall_s"])
        except (OSError, ValueError, KeyError):
            continue
        if run_id.startswith(TEACHER_PREFIXES):
            teacher += wall
        elif run_id in DEV_RUNS:
            dev += wall
        else:
            b = block_of(run_id, blocks)
            by_block[b] = by_block.get(b, 0.0) + wall
    by_block_h = {b: s / 3600 for b, s in sorted(by_block.items())}
    return {
        "teacher_h": teacher / 3600,
        "teacher_cap_h": teacher_cap_h,
        "experiment_h": sum(by_block_h.values()),
        "dev_h": dev / 3600,
        "by_block": by_block_h,
    }


def write(
    runs: Path = RUNS_DIR, out: Path = LEDGER_PATH, blocks: set[str] | None = None
) -> dict[str, Any]:
    d = compute(runs, blocks)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2) + "\n")
    return d


if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
