#!/usr/bin/env bash
# Block until the GPU has been idle (< ${MAX_MIB:-2500} MiB used) for ${IDLE_MIN:-10} consecutive
# minutes. Use before a queue when another project (e.g. an ollama bench) shares the GPU:
# ollama unloads models between calls, so a single idle reading is not enough.
set -uo pipefail
need=$(( ${IDLE_MIN:-10} * 2 )); ok=0
while (( ok < need )); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( used < ${MAX_MIB:-2500} )); then ok=$((ok + 1)); else ok=0; fi
  sleep 30
done
echo "$(date -Is) GPU idle for ${IDLE_MIN:-10} min"
