"""Expand config/grid.yaml into cells (docs/specs/experiments.md; MISSION §4).

This module is the single source of truth for what runs and what it costs: ``budget.py`` and
``run.py`` both import it, so the projection and the launched queue can never disagree. It reads no
result or metric files (only the grid, ``results/throughput.json`` and ``base_revisions.json``).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

GRID_PATH = Path("config/grid.yaml")
THROUGHPUT_PATH = Path("results/throughput.json")
REVISIONS_PATH = Path("results/base_revisions.json")

# Per-run inference units (D-034): test_perturbed reuses test predictions for units without numbers,
# mr_pool = numeric units of non-train splits; silver_val = 1k selection subset.
EVAL_UNITS = {
    "test": 4407, "test_perturbed": 440, "mr_pool": 689, "heldout": 641, "silver_val": 1000,
    "gold": 300,
}  # fmt: skip
QUANT_EVAL_UNITS = 2129  # 1,000 test subset + 1,129 numeric pool
GRPO_DEFAULT_H = 2.0
QUANT_EXPORT_H = 0.3
SLOWDOWN = {"full": 1.8, "qlora": 1.4}  # vs LoRA bf16 throughput
TEACHER_BASES = {"qwen3-14b-awq": "Qwen/Qwen3-14B-AWQ"}  # zero-shot only; not in `bases`
TRAIN_METHODS = ("lora", "qlora", "full")
GGUF = ("gguf-q8", "gguf-q4km")

# docs/RESEARCH_QUESTIONS.md §7: further cuts, only in this order.
CUT_ORDER: list[tuple[str, str]] = [
    ("drop_block8", "drop block 8 (thai_optional)"),
    ("thai_1seed", "Thai cells (block 7) to 1 seed"),
    ("drop_gptq", "drop GPTQ-4 where AWQ-4 exists"),
    ("drop_4b_cpu", "drop 4B CPU/llama.cpp runs"),
    ("n20k_4b_1seed", "N = 20k for 4B to 1 seed"),
    ("8b_labeleff_1seed", "8B label-efficiency cells to 1 seed"),
    ("method_variants_1seed", "method_variants to 1 seed"),
]


class Cell(BaseModel):
    block: str
    priority: int
    base: str  # grid key, e.g. qwen3-1.7b
    hf: str
    revision: str  # "" when the base has no pin in base_revisions.json (not launchable)
    method: str  # zero_shot | lora | qlora | full | grpo | quant
    n: int | None = None
    arm: str = "label"  # label | rationale | label_token_matched | placebo
    seed: int = 0
    train_langs: list[str] = Field(default_factory=lambda: ["en", "th"])
    eval_sets: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    hours: float = 0.0  # projected GPU-hours for this cell (train + its evaluation)

    @property
    def pinned(self) -> bool:
        return bool(self.revision)

    @property
    def run_id(self) -> str:
        if self.method == "zero_shot":
            return f"zs-{self.base}"
        if self.method == "grpo":
            return f"grpo-{self.base}-s{self.seed}"
        if self.method == "quant":
            return f"q-{self.base}-{self.extra['format']}-{self.extra['backend']}"
        langs = "_".join(self.train_langs)
        return f"{self.block}-{self.base}-{self.method}-n{self.n}-{self.arm}-{langs}-s{self.seed}"


def as_list(x: object) -> list[Any]:
    return x if isinstance(x, list) else [x]


def load_grid(path: Path = GRID_PATH) -> dict[str, Any]:
    g: dict[str, Any] = yaml.safe_load(path.read_text())
    return g


def load_throughput(path: Path = THROUGHPUT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def load_revisions(path: Path = REVISIONS_PATH) -> dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}


class Rates:
    """Measured-over-assumed throughputs."""

    def __init__(self, g: dict[str, Any], meas: dict[str, Any]) -> None:
        self.tok_s = {**g["assumed_train_tok_s"], **meas.get("train_tok_s", {})}
        self.inf = {**g["assumed_infer_units_h"], **meas.get("infer_units_h", {})}
        self.tpe = meas.get("tokens_per_example", g["tokens_per_example"])

    def eval_h(self, base: str, sets: list[str]) -> float:
        return sum(EVAL_UNITS[s] for s in sets) / self.inf[base]

    def train_h(self, base: str, method: str, n: int, arm: str) -> float:
        n_eff = n * 1.46  # + rule-empty units at the train ratio (D-027)
        epochs = epochs_for(n)
        examples = max(epochs * n_eff, 300 * 16)
        tokens = examples * self.tpe * (1.6 if arm == "rationale" else 1.0)
        return tokens / self.tok_s[base] / 3600 * SLOWDOWN.get(method, 1.0)


def epochs_for(n: int) -> int:
    return 1 if n >= 10000 else 3  # D-035


def quant_valid(backend: str, fmt: str) -> bool:
    """vLLM serves the non-GGUF formats; llama.cpp (CUDA or CPU) serves GGUF only."""
    return fmt in GGUF if backend.startswith("llamacpp") else fmt not in GGUF


def expand(
    g: dict[str, Any],
    meas: dict[str, Any] | None = None,
    revisions: dict[str, str] | None = None,
    cuts: frozenset[str] | set[str] = frozenset(),
    strict: bool = False,
) -> list[Cell]:
    """All cells of the grid in priority order (stable within a priority), cuts applied.

    A base without a pinned revision raises when ``strict``; otherwise its cells carry
    ``revision == ""`` (projectable, but the runner refuses to emit them)."""
    rates = Rates(g, meas or {})
    revs = load_revisions() if revisions is None else revisions
    hf_of = {k: v["hf"] for k, v in g["bases"].items()} | TEACHER_BASES
    cells: list[Cell] = []

    def mk(b: dict[str, Any], base: str, method: str, hours: float, **kw: Any) -> Cell:
        hf = hf_of[base]
        if hf not in revs and strict:
            raise KeyError(f"no pinned revision for {hf} in {REVISIONS_PATH}; refusing to expand")
        return Cell(
            block=b["name"], priority=b["priority"], base=base, hf=hf, revision=revs.get(hf, ""),
            method=method, hours=hours, **kw,
        )  # fmt: skip

    for b in sorted(g["blocks"], key=lambda b: b["priority"]):  # sorted() is stable
        for c in b["cells"]:
            sets = c.get("eval_sets", g["eval_sets"])
            langs = c.get("train_langs", g["train_langs"])
            seeds = g["seeds"] if c.get("seeds") == "all" else c.get("seeds", g["seeds"])
            methods = as_list(c.get("method", "quant" if b["name"] == "quantization" else "lora"))
            for base, method in itertools.product(as_list(c["base"]), methods):
                if method == "zero_shot":
                    cells.append(
                        mk(b, base, method, rates.eval_h(base, sets), eval_sets=sets, seed=0,
                           n=None, train_langs=[], extra={"student_prompt": True})
                    )  # fmt: skip
                elif method == "quant":
                    cells += _quant_cells(b, c, base, rates, mk)
                elif method == "grpo":
                    for s in seeds:
                        mxw = c.get("max_wall_h", GRPO_DEFAULT_H)
                        cells.append(
                            mk(b, base, method, mxw + rates.eval_h(base, sets), seed=s,
                               eval_sets=sets, train_langs=langs,
                               extra={"max_wall_h": mxw, "from": c.get("from", "best_sft")})
                        )  # fmt: skip
                else:
                    if (
                        method == "qlora"
                        and base != "qwen3-8b"
                        and b["name"] != "method_variants"
                        and {"lora", "qlora"} <= set(methods)
                    ):
                        continue  # a [lora, qlora] cell: qlora only for 8B
                    if method == "lora" and base == "qwen3-8b":
                        continue
                    for n in as_list(c.get("n", 10000)):
                        for s in seeds:
                            h = rates.train_h(base, method, n, c.get("arm", "label"))
                            cells.append(
                                mk(b, base, method, h + rates.eval_h(base, sets), n=n,
                                   arm=c.get("arm", "label"), seed=s, train_langs=langs,
                                   eval_sets=sets)
                            )  # fmt: skip
    return apply_cuts(cells, set(cuts))


def _quant_cells(
    b: dict[str, Any], c: dict[str, Any], base: str, rates: Rates, mk: Any
) -> list[Cell]:
    """One cell per valid (format, backend); the base's hours are spread so the sum is exact."""
    fmts, backends = c["formats"], c["backends"]
    units = c.get("eval_units", QUANT_EVAL_UNITS)
    gpu = [(f, be) for f in fmts for be in backends if quant_valid(be, f)]
    cpu = [(f, "llamacpp-cpu") for f in fmts if f in GGUF] if base in c.get("cpu_bases", []) else []
    gpu_h = QUANT_EXPORT_H + len(fmts) * units / rates.inf[base] * 1.5
    cpu_h = 300 / (rates.inf[base] / 15) if base in c.get("cpu_bases", []) else 0.0
    if not cpu:  # nothing to carry the CPU time: fold it into the GPU cells
        gpu_h, cpu_h = gpu_h + cpu_h, 0.0
    out = []
    for f, be in gpu + cpu:
        h = gpu_h / len(gpu) if (f, be) in gpu else cpu_h / len(cpu)
        out.append(mk(b, base, "quant", h, n=None, seed=0, train_langs=[],
                      eval_sets=["test"], extra={"format": f, "backend": be, "from": c.get("from"),
                                                 "eval_units": units}))  # fmt: skip
    return out


def apply_cuts(cells: list[Cell], cuts: set[str]) -> list[Cell]:
    """Apply pre-registered §7 cuts (see CUT_ORDER) to expanded cells."""

    def keep(x: Cell) -> bool:
        if "drop_block8" in cuts and x.priority == 8:
            return False
        if "thai_1seed" in cuts and x.block == "thai" and x.seed > 0:
            return False
        if "drop_gptq" in cuts and x.method == "quant" and x.extra["format"] == "gptq":
            return False
        if (
            "drop_4b_cpu" in cuts
            and x.method == "quant"
            and x.base == "qwen3-4b"
            and x.extra["backend"] == "llamacpp-cpu"
        ):
            return False
        if "n20k_4b_1seed" in cuts and x.base == "qwen3-4b" and x.n == 20000 and x.seed > 0:
            return False
        if (
            "8b_labeleff_1seed" in cuts
            and x.block == "label_efficiency"
            and x.base == "qwen3-8b"
            and x.seed > 0
        ):
            return False
        return not ("method_variants_1seed" in cuts and x.block == "method_variants" and x.seed > 0)

    return [x for x in cells if keep(x)]


def block_summary(cells: list[Cell]) -> list[tuple[int, str, int, float]]:
    """(priority, block, runs, gpu_h) in priority order."""
    out: dict[tuple[int, str], list[float]] = {}
    for x in cells:
        out.setdefault((x.priority, x.block), []).append(x.hours)
    return [(p, n, len(v), sum(v)) for (p, n), v in out.items()]
