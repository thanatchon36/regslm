#!/usr/bin/env bash
# Run a long job detached in tmux with a 1 Hz GPU trace and a DONE marker.
# usage: scripts/tmux_job.sh <run_id> <command...>
set -euo pipefail
RUN_ID="$1"; shift
DIR="results/runs/${RUN_ID}"
mkdir -p "$DIR"
if [[ -f "$DIR/DONE" ]]; then echo "skip: $RUN_ID already DONE"; exit 0; fi
if tmux has-session -t "$RUN_ID" 2>/dev/null; then echo "already running: $RUN_ID"; exit 1; fi
CMD=$(printf '%q ' "$@")
tmux new-session -d -s "$RUN_ID" bash -c "
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > '$DIR/gpu.csv' 2>/dev/null & GPU=\$!
  start=\$(date +%s)
  ( $CMD ) > '$DIR/train.log' 2>&1; rc=\$?
  kill \$GPU 2>/dev/null
  echo \"{\\\"run_id\\\": \\\"$RUN_ID\\\", \\\"exit_code\\\": \$rc, \\\"wall_s\\\": \$(( \$(date +%s) - start ))}\" > '$DIR/exit.json'
  [[ \$rc -eq 0 ]] && touch '$DIR/DONE'
"
echo "launched $RUN_ID in tmux (attach: tmux attach -t $RUN_ID; log: $DIR/train.log)"
