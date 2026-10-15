#!/usr/bin/env bash
# Run GPU jobs strictly one after another. Each line of the queue file:
#   <make target> <RUN id> [ARGS...]      e.g.  label gold-pass-a --units X --variant gold_a
# Waits for results/runs/<RUN>/exit.json before starting the next; skips runs already DONE.
set -uo pipefail
QUEUE="$1"
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  read -r target run args <<<"$line"
  dir="results/runs/$run"
  if [[ -f "$dir/DONE" ]]; then echo "$(date -Is) skip $run (DONE)"; continue; fi
  while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 2500 ]]; do sleep 30; done
  rm -f "$dir/exit.json"
  echo "$(date -Is) start $run"
  make "$target" RUN="$run" ARGS="$args"
  until [[ -f "$dir/exit.json" ]]; do sleep 30; done
  echo "$(date -Is) end $run $(cat "$dir/exit.json")"
done < "$QUEUE"
