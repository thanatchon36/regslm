"""Experiment runner: grid cells -> configs + queue files (docs/specs/experiments.md).

``make experiments [BLOCK=name] [DRY=1]`` (= ``python -m experiments.run``). Blocks are processed
in priority order. Before emitting a block the budget rule runs (``budget.decide``: logged GPU-h +
remaining projection of this and all higher-priority blocks <= ``budget_gpu_hours``); if it fails
the pre-registered cuts that would be needed are printed and nothing more is emitted (exit 3).
Emitted: ``results/runs/<id>/config.yaml`` (+ ``grpo.yaml`` / ``zero_shot.yaml``) and
``results/queues/<block>.txt`` lines in ``scripts/run_queue.sh`` format (idempotent; DONE stages
are skipped). Run the queue with ``make queue-run QUEUE=results/queues/<block>.txt``.

Queue stage markers (all via ``scripts/tmux_job.sh``, so each has gpu.csv / exit.json / DONE):
  train/grpo   results/runs/<id>/            predict   results/runs/<id>/predict/
  evaluate     results/runs/<id>/evaluate/   quantize  results/runs/<parent>-quantize/
  bench-cell   results/runs/q-<base>-<format>-<backend>/
``run_queue.sh`` derives the marker directory from the target name (see its header).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from experiments import budget, ledger
from experiments import grid as G
from regslm.train.config import TrainConfig
from regslm.train.predict import SET_NAMES

RUNS = Path("results/runs")
QUEUES = Path("results/queues")
RATIONALE_LABELS = "results/runs/silver-train-rationale/labels.jsonl"
DEFAULT_LANGS = ["en", "th"]
EXIT_OVER_BUDGET = 3


@dataclass
class Stage:
    target: str  # make target
    run: str  # RUN= value as written in the queue file
    args: str
    marker: Path  # DONE file of this stage

    @property
    def line(self) -> str:
        return " ".join(x for x in (self.target, self.run, self.args) if x)


def runnable_sets(sets: Sequence[str]) -> tuple[list[str], list[str]]:
    ok = [s for s in sets if s in SET_NAMES]
    return ok, [s for s in sets if s not in SET_NAMES]


def stages(cell: G.Cell, runs: Path = RUNS, parent: str | None = None) -> list[Stage]:
    rid = cell.run_id
    d = runs / rid
    if cell.method == "quant":
        be, fm = cell.extra["backend"], cell.extra["format"]
        assert parent is not None
        return [
            Stage("quantize", parent, "", runs / f"{parent}-quantize" / "DONE"),
            Stage("bench-cell", rid, f"{parent} {be} {fm}", d / "DONE"),
        ]
    sets = ",".join(runnable_sets(cell.eval_sets)[0])
    ev = [
        Stage("predict", rid, f"--sets {sets}", d / "predict" / "DONE"),
        Stage("evaluate", rid, f"--sets {sets}", d / "evaluate" / "DONE"),
    ]
    if cell.method == "zero_shot":
        return ev
    return [Stage("grpo" if cell.method == "grpo" else "train", rid, "", d / "DONE"), *ev]


def cell_done(cell: G.Cell, runs: Path = RUNS) -> bool:
    d = runs / cell.run_id
    if cell.method == "quant":
        return (d / "DONE").exists()
    return (d / "evaluate" / "DONE").exists()


def write_config(cell: G.Cell, runs: Path = RUNS, parent: str | None = None) -> None:
    """Write run config(s), identical to ``regslm train-config`` / ``grpo-config`` output."""
    rid = cell.run_id
    if cell.method == "grpo":
        from regslm.train.grpo import write_run_config

        assert parent is not None
        write_run_config(
            rid, parent, {"seed": cell.seed, "max_wall_s": cell.extra["max_wall_h"] * 3600}, runs
        )
    elif cell.method == "zero_shot":
        # Student prompt for every base, incl. the 14B teacher model (D-0xx): no checkpoint.
        TrainConfig(run_id=rid, base=cell.hf, revision=cell.revision, n_labels=0).save(runs)
        (runs / rid / "zero_shot.yaml").write_text(
            yaml.safe_dump({"model": cell.hf, "revision": cell.revision, "prompt": "student"})
        )
    elif cell.method == "quant":
        return  # operates on the parent's merged checkpoint
    else:
        d: dict[str, object] = {
            "run_id": rid, "base": cell.hf, "revision": cell.revision, "method": cell.method,
            "n_labels": cell.n, "arm": cell.arm, "seed": cell.seed,
            "epochs": G.epochs_for(cell.n or 0),
        }  # fmt: skip
        if cell.train_langs != DEFAULT_LANGS:
            d["langs"] = cell.train_langs
        if cell.arm == "label_token_matched":
            d.update(arm="label", match_rationale_tokens=True, rationale_labels=[RATIONALE_LABELS])
        elif cell.arm == "rationale":
            d["rationale_labels"] = [RATIONALE_LABELS]
        TrainConfig.model_validate(d).save(runs)


def resolve_parent(cell: G.Cell, runs: Path = RUNS) -> str | None:
    """Best evaluated SFT run (silver_val OF1) of this base, or None if none exists yet."""
    from regslm.train.grpo import select_parent

    try:
        return select_parent(cell.hf, runs)
    except LookupError:
        return None


def append_lines(path: Path, lines: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    have = path.read_text().splitlines() if path.exists() else []
    new = [ln for ln in lines if ln not in have]
    if new or not path.exists():
        with path.open("a") as f:
            if not have:
                f.write(f"# {path.stem}: generated by experiments/run.py (idempotent)\n")
            for ln in new:
                f.write(ln + "\n")
    return len(new)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--block", help="only this block (budget check still counts higher priority)")
    ap.add_argument("--dry", action="store_true", help="print cells, run ids and hours only")
    ap.add_argument("--grid", type=Path, default=G.GRID_PATH)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--queues", type=Path, default=QUEUES)
    a = ap.parse_args(argv)

    g = G.load_grid(a.grid)
    meas = G.load_throughput()
    cells = G.expand(g, meas)
    names = [b["name"] for b in sorted(g["blocks"], key=lambda b: b["priority"])]
    if a.block and a.block not in names:
        print(f"unknown block {a.block!r}; choose from {names}", file=sys.stderr)
        return 2
    logged = ledger.compute(a.runs, set(names))
    logged_h = logged["experiment_h"]
    print(
        f"budget {g['budget_gpu_hours']} GPU-h; logged experiment {logged_h:.2f} h; "
        f"teacher {logged['teacher_h']:.2f}/{g['teacher_gpu_hours']} h (separate)"
    )
    done = lambda c: cell_done(c, a.runs)  # noqa: E731
    cum = 0.0
    parents: dict[str, str | None] = {}
    for blk in names:
        bc = [c for c in cells if c.block == blk]
        cum += sum(c.hours for c in bc)
        if a.block and blk != a.block:
            continue
        prio = bc[0].priority
        todo = [c for c in bc if not done(c)]
        print(
            f"\n[{prio}] {blk}: {len(bc)} runs, {sum(c.hours for c in bc):.1f} GPU-h projected "
            f"({len(todo)} to do; cumulative all blocks {cum:.1f})"
        )
        for c in bc:
            tag = "DONE" if done(c) else ("" if c.pinned else "UNPINNED")
            print(f"  {c.run_id:<62s} {c.hours:6.2f} h  {tag}")
        dec = budget.decide(g, prio, logged_h, done, meas)
        status = "OK" if dec.ok else "OVER BUDGET"
        print(
            f"  budget check: logged {dec.logged_h:.1f} + remaining(priority<={prio}) "
            f"{dec.projected_h:.1f} = {dec.logged_h + dec.projected_h:.1f} / {dec.budget_h:.0f}"
            f" -> {status}"
        )
        if not dec.ok:
            print("  pre-registered cuts needed (RESEARCH_QUESTIONS §7 order):")
            for ln in budget.describe_cuts(dec.cuts):
                print(f"    {ln}")
            tail = "" if dec.sufficient else " (still over budget after all cuts)"
            print(f"    -> projection after cuts {dec.logged_h + dec.cut_projected_h:.1f}{tail}")
            if not a.dry:
                print(f"  not emitting block {blk}; edit config/grid.yaml per the cuts above")
                return EXIT_OVER_BUDGET
        if a.dry:
            continue
        lines: list[str] = []
        for c in todo:
            if not c.pinned:
                print(f"  refuse {c.run_id}: no pinned revision for {c.hf}", file=sys.stderr)
                return 2
            parent = None
            if c.method in ("grpo", "quant"):
                parent = parents.setdefault(c.hf, resolve_parent(c, a.runs))
                if parent is None:
                    print(f"  defer {c.run_id}: no evaluated SFT parent for {c.hf} yet")
                    continue
            dropped = runnable_sets(c.eval_sets)[1]
            if dropped:
                print(f"  note {c.run_id}: sets not supported by predict yet: {dropped}")
            write_config(c, a.runs, parent)
            lines += [s.line for s in stages(c, a.runs, parent) if not s.marker.exists()]
        qf = a.queues / f"{blk}.txt"
        n = append_lines(qf, lines)
        print(f"  {qf}: +{n} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
