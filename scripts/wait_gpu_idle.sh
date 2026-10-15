#!/usr/bin/env bash
# Block until the GPU has been idle (< ${MAX_MIB:-2500} MiB used) for ${IDLE_MIN:-10} consecutive
# minutes, and (if WAIT_PROC is set) no process matches the pgrep pattern WAIT_PROC. Use before a
# queue when another project (e.g. an ollama bench) shares the GPU: ollama unloads models between
# calls, so a single idle reading is not enough. Write WAIT_PROC as e.g. "[h]arnesswatch bench" so the
# pattern does not match the command line that launched this script.
set -uo pipefail
need=$(( ${IDLE_MIN:-10} * 2 )); ok=0
while (( ok < need )); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  busy=0; [[ -n "${WAIT_PROC:-}" ]] && pgrep -f "$WAIT_PROC" > /dev/null && busy=1
  if (( used < ${MAX_MIB:-2500} && busy == 0 )); then ok=$((ok + 1)); else ok=0; fi
  sleep 30
done
echo "$(date -Is) GPU idle for ${IDLE_MIN:-10} min"
