"""Project GPU-hours for config/grid.yaml (MISSION §4 budget rule).

Uses measured throughput from results/throughput.json when present, else the grid's assumed
values. Prints a per-block table and the cumulative total in priority order.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

# Per-run inference units (D-034): test_perturbed reuses test predictions for units without numbers,
# mr_pool = numeric units of non-train splits; silver_val = 1k selection subset.
EVAL_UNITS = {"test": 4407, "test_perturbed": 440, "mr_pool": 689, "heldout": 641, "silver_val": 1000, "gold": 300}
QUANT_EVAL_UNITS = 2129  # 1,000 test subset + 1,129 numeric pool
GRPO_DEFAULT_H = 2.0
QUANT_EXPORT_H = 0.3


def as_list(x: object) -> list:
    return x if isinstance(x, list) else [x]


def main() -> None:
    g = yaml.safe_load(Path("config/grid.yaml").read_text())
    meas = json.loads(Path("results/throughput.json").read_text()) if Path("results/throughput.json").exists() else {}
    tok_s = {**g["assumed_train_tok_s"], **meas.get("train_tok_s", {})}
    inf = {**g["assumed_infer_units_h"], **meas.get("infer_units_h", {})}
    tpe = meas.get("tokens_per_example", g["tokens_per_example"])
    seeds_all = g["seeds"]

    def eval_h(base: str, sets: list[str]) -> float:
        return sum(EVAL_UNITS[s] for s in sets) / inf[base]

    def train_h(base: str, method: str, n: int, arm: str) -> float:
        n_eff = n * 1.46  # + rule-empty units at the train ratio (D-027)
        epochs = 1 if n >= 10000 else 3  # D-035
        examples = max(epochs * n_eff, 300 * 16)
        tokens = examples * tpe * (1.6 if arm == "rationale" else 1.0)
        slow = {"full": 1.8, "qlora": 1.4}.get(method, 1.0)
        return tokens / tok_s[base] / 3600 * slow

    total = 0.0
    for b in sorted(g["blocks"], key=lambda b: b["priority"]):
        bh = 0.0
        runs = 0
        for c in b["cells"]:
            sets = c.get("eval_sets", g["eval_sets"])
            for base, method in itertools.product(as_list(c["base"]), as_list(c.get("method", "lora"))):
                if method == "zero_shot":
                    bh += eval_h(base, sets); runs += 1; continue
                if b["name"] == "quantization":
                    n_fmt = len(c["formats"])
                    bh += QUANT_EXPORT_H + n_fmt * QUANT_EVAL_UNITS / inf[base] * 1.5
                    if base in c.get("cpu_bases", []):
                        bh += 300 / (inf[base] / 15)  # CPU-only llama.cpp, ~15x slower
                    runs += n_fmt; continue
                seeds = seeds_all if c.get("seeds") == "all" else c.get("seeds", seeds_all)
                if method == "grpo":
                    bh += len(seeds) * (c.get("max_wall_h", GRPO_DEFAULT_H) + eval_h(base, sets)); runs += len(seeds); continue
                if method == "qlora" and base != "qwen3-8b" and b["name"] != "method_variants" and "qlora" in as_list(c.get("method")) and "lora" in as_list(c.get("method")) and base != "qwen3-8b":
                    continue  # label_efficiency lists [lora, qlora]: qlora only for 8B
                if method == "lora" and base == "qwen3-8b":
                    continue
                for n in as_list(c.get("n", 10000)):
                    bh += len(seeds) * (train_h(base, method, n, c.get("arm", "label")) + eval_h(base, sets))
                    runs += len(seeds)
        total += bh
        print(f"{b['priority']} {b['name']:18s} runs={runs:3d} gpu_h={bh:6.1f} cumulative={total:6.1f}")
    print(f"TOTAL {total:.1f} / {g['budget_gpu_hours']} GPU-h")


if __name__ == "__main__":
    main()
