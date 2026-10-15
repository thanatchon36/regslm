#!/usr/bin/env bash
# Apple-M4 (or any Mac) llama.cpp benchmark; the owner runs this, we ingest the JSON.
# usage: scripts/bench_mac.sh <model.gguf> [label] [prompts.jsonl]
#   prompts.jsonl: one {"messages": [...]} per line (export with `regslm bench ... --limit`
#   or use the default 3 short prompts). Needs: brew install llama.cpp  (llama-server on PATH)
# Output: bench_mac-<label>.json  {chip, threads, n, gen_tok_per_s, latency p50/p95, rss_mib}
set -euo pipefail
GGUF="${1:?model.gguf}"; LABEL="${2:-$(basename "$GGUF" .gguf)}"; PROMPTS="${3:-}"
PORT=18080; OUT="bench_mac-${LABEL}.json"
llama-server -m "$GGUF" --host 127.0.0.1 --port $PORT --n-gpu-layers 999 --ctx-size 4096 \
  --jinja --reasoning off --no-webui >/tmp/llama-server-mac.log 2>&1 &
PID=$!; trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
python3 - "$PORT" "$PID" "$OUT" "$LABEL" "$PROMPTS" <<'PY'
import json, platform, subprocess, sys, time, urllib.request
port, pid, out, label, prompts = sys.argv[1:6]
if prompts:
    msgs = [json.loads(l)["messages"] for l in open(prompts) if l.strip()]
else:
    msgs = [[{"role": "user", "content": q}] for q in (
        "List two duties of a bank under capital rules.",
        "Summarise: Institutions shall report large exposures within 10 days.",
        "What is a leverage ratio?")]
def chat(m):
    body = json.dumps({"model": "m", "messages": m, "temperature": 0, "max_tokens": 512,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})))
    return time.perf_counter() - t, r["usage"]["completion_tokens"]
chat(msgs[0])  # warm-up
res = [chat(m) for m in msgs]
lat = sorted(r[0] for r in res); gen = sum(r[1] for r in res)
rss = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip()
chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
json.dump({"label": label, "backend": "llamacpp-metal", "chip": chip or platform.processor(),
           "n": len(res), "gen_tok_per_s": gen / sum(lat),
           "latency_s": {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))]},
           "memory": {"host_ram_peak_mib": int(rss or 0) / 1024}}, open(out, "w"), indent=2)
print(open(out).read())
PY
