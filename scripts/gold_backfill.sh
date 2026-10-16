#!/usr/bin/env bash
# Add the gold set to finished runs (D-044: gold was pending when they were predicted), then
# re-evaluate all their prediction files. Usage: scripts/gold_backfill.sh RUN_ID...
# GPU job: run it only when no other GPU job is running (e.g. after the block queue ends).
# Log: results/gold_backfill.log. Skips runs that already have predictions/gold.jsonl.
set -uo pipefail
UV="uv run --no-sync"
log() { echo "$(date -Is) $*" >> results/gold_backfill.log; }
for run in "$@"; do
  if [[ ! -f "results/runs/$run/DONE" ]]; then log "skip $run (not trained)"; continue; fi
  if [[ -f "results/runs/$run/predictions/gold.jsonl" ]]; then log "skip $run (has gold)"; continue; fi
  $UV regslm predict --run-id "$run" --sets gold >> results/gold_backfill.log 2>&1; rc=$?
  log "predict gold $run exit $rc"
  [[ $rc -eq 0 ]] || continue
  $UV regslm evaluate --run-id "$run" >> results/gold_backfill.log 2>&1; log "evaluate $run exit $?"
done
