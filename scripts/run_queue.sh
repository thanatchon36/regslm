#!/usr/bin/env bash
# Run GPU jobs strictly one after another. Each line of the queue file:
#   <make target> <RUN id> [ARGS...]      e.g.  label gold-pass-a --units X --variant gold_a
# Marker directory (results/runs/<key>/) is <RUN> except: predict -> <RUN>/predict, evaluate ->
# <RUN>/evaluate, quantize -> <RUN>-quantize (matches the Makefile's tmux_job.sh run ids).
# Waits for results/runs/<key>/exit.json before starting the next; skips runs already DONE.
set -uo pipefail
QUEUE="$1"
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  read -r target run args <<<"$line"
  key="$run"
  case "$target" in
    predict|evaluate) key="$run/$target" ;;
    quantize) key="$run-quantize" ;;
  esac
  dir="results/runs/$key"
  if [[ -f "$dir/DONE" ]]; then echo "$(date -Is) skip $key (DONE)"; continue; fi
  while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -gt 2500 ]]; do sleep 30; done
  rm -f "$dir/exit.json"
  echo "$(date -Is) start $key"
  make "$target" RUN="$run" ARGS="$args"
  until [[ -f "$dir/exit.json" ]]; do sleep 30; done
  echo "$(date -Is) end $key $(cat "$dir/exit.json")"
done < "$QUEUE"
