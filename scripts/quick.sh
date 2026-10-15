#!/usr/bin/env bash
# End-to-end smoke: 200-row fixture -> Qwen3-0.6B LoRA (50 steps) -> predict 50 units -> evaluate.
# Run through scripts/tmux_job.sh (make quick). Wall times go to results/timing.json.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
RUN=quick
UVC="uv run --no-sync"
UVT="env UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync"
t0=$(date +%s)
$UVT regslm train-config --run-id $RUN --base Qwen/Qwen3-0.6B --method lora --n-labels 1000 \
  --set 'labels=["tests/fixtures/quick_labels.jsonl"]' --set splits=null --set eval_n=50 \
  --set max_steps=50 --set min_steps=0 --set rule_empty=false
$UVT regslm train --run-id $RUN
t1=$(date +%s)
$UVC regslm predict --run-id $RUN --sets quick --limit 50
t2=$(date +%s)
$UVC regslm evaluate --run-id $RUN --sets quick --n-boot 200
t3=$(date +%s)
$UVC python - "$t0" "$t1" "$t2" "$t3" <<'PY'
import json, sys
from pathlib import Path
t0, t1, t2, t3 = map(int, sys.argv[1:5])
p = Path("results/timing.json")
d = json.loads(p.read_text()) if p.exists() else {}
d["quick"] = {"wall_s": t3 - t0, "train_s": t1 - t0, "predict_s": t2 - t1, "evaluate_s": t3 - t2}
p.write_text(json.dumps(d, indent=2))
print(d["quick"])
PY
