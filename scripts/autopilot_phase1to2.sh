#!/usr/bin/env bash
# Chains the CPU steps and the first experiment blocks after the teacher queue (D-045).
# Each step waits for its inputs; every step's outcome is logged to results/autopilot.log.
# Stops after block 2 so that throughput re-projection (RESEARCH_QUESTIONS §7) is done by hand.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
UV="uv run --no-sync"
log() { echo "$(date -Is) $*" >> results/autopilot.log; }
wait_for() { until [[ -f "$1" ]]; do sleep 60; done; }

wait_for results/runs/silver-eval-all-label/DONE
log "eval labels done -> eval-sets"
$UV regslm eval-sets --labels results/runs/silver-eval-all-label/labels.jsonl >> results/autopilot.log 2>&1; log "eval-sets exit $?"

wait_for results/runs/gold-pass-b/exit.json
log "gold passes done -> gold build"
$UV regslm gold build >> results/autopilot.log 2>&1; log "gold build exit $?"

wait_for results/runs/silver-train-rationale/exit.json
log "teacher queue finished"
for blk in zero_shot sft_ladder_10k placebo; do
  make experiments BLOCK=$blk >> results/autopilot.log 2>&1; rc=$?
  log "emit $blk exit $rc"
  if [[ $rc -ne 0 ]]; then log "STOP: emit failed or budget gate (exit 3)"; exit $rc; fi
  if [[ -f results/queues/$blk.txt ]]; then
    scripts/run_queue.sh results/queues/$blk.txt >> results/queue.log 2>&1
    log "queue $blk finished"
  fi
done
log "autopilot done (blocks 1-2); re-project budget before block 3"
