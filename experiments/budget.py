"""GPU-hour projection and the pre-committed budget rule (docs/RESEARCH_QUESTIONS.md §7).

Uses measured throughput from results/throughput.json when present, else the grid's assumed
values (all expansion logic lives in ``experiments/grid.py``). Decision code here reads only the
grid, throughput, pinned revisions, ``exit.json`` wall times (via ``ledger``) and DONE markers —
never metrics.json / eval.json. ``python -m experiments.budget`` prints the per-block table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from experiments import grid as G


@dataclass
class Decision:
    ok: bool
    logged_h: float
    projected_h: float  # remaining cells of this block and all higher-priority blocks
    budget_h: float
    cuts: list[str] = field(default_factory=list)  # cumulative §7 cuts that would be needed
    sufficient: bool = True  # False: even all cuts leave the projection over budget
    cut_projected_h: float = 0.0


def remaining_hours(cells: list[G.Cell], priority: int, is_done: Callable[[G.Cell], bool]) -> float:
    return sum(c.hours for c in cells if c.priority <= priority and not is_done(c))


def decide(
    g: dict[str, Any],
    priority: int,
    logged_h: float,
    is_done: Callable[[G.Cell], bool],
    meas: dict[str, Any] | None = None,
    revisions: dict[str, str] | None = None,
) -> Decision:
    """Budget check before launching a block of ``priority`` (cap is not soft)."""
    cap = float(g["budget_gpu_hours"])
    cells = G.expand(g, meas, revisions)
    proj = remaining_hours(cells, priority, is_done)
    if logged_h + proj <= cap:
        return Decision(True, logged_h, proj, cap)
    applied: list[str] = []
    cut_proj = proj
    for key, _ in G.CUT_ORDER:
        applied.append(key)
        cut_proj = remaining_hours(G.expand(g, meas, revisions, set(applied)), priority, is_done)
        if logged_h + cut_proj <= cap:
            return Decision(False, logged_h, proj, cap, applied, True, cut_proj)
    return Decision(False, logged_h, proj, cap, applied, False, cut_proj)


def describe_cuts(keys: list[str]) -> list[str]:
    text = dict(G.CUT_ORDER)
    return [f"({i}) {text[k]}" for i, k in enumerate(keys, 1)]


def main() -> None:
    g = G.load_grid()
    cells = G.expand(g, G.load_throughput())
    total = 0.0
    for prio, name, runs, h in G.block_summary(cells):
        total += h
        print(f"{prio} {name:18s} runs={runs:3d} gpu_h={h:6.1f} cumulative={total:6.1f}")
    print(f"TOTAL {total:.1f} / {g['budget_gpu_hours']} GPU-h")


if __name__ == "__main__":
    main()
