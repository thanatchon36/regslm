"""RQ4 table: aggregate results/bench/<run>/*.json (docs/specs/quant_bench.md).

usage: python experiments/rq4_quant_table.py <run> [more runs ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from regslm.bench.aggregate import collect, to_markdown

if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(f"## {run}\n")
        print(to_markdown(collect(Path("results/bench") / run)), "\n")
