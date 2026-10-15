# Spec: `regslm.quant` + `regslm.bench` (paper §4.6, RQ4)

## quant (export; record tool versions in `results/runs/<id>/quant.json`)
From a merged bf16 checkpoint (`results/runs/<sft-or-grpo-id>/checkpoints/merged`):
- **FP8**: `llmcompressor` (if installable in `.venv` without changing vLLM/torch; else a
  separate `.venv-quant`) `FP8_DYNAMIC` scheme (W8A8 dynamic per-token) → `quant/fp8/`.
  Also vLLM on-the-fly `quantization="fp8"` as a no-export baseline.
- **AWQ-4**: llmcompressor AWQ W4A16 (group 128), calibration = 256 *train* units rendered with
  the student prompt (never test/gold) → `quant/awq/`.
- **GPTQ-4**: llmcompressor GPTQ W4A16 g128, same calibration set → `quant/gptq/`.
- **GGUF**: llama.cpp `convert_hf_to_gguf.py` → f16 GGUF → `llama-quantize` Q8_0 and Q4_K_M →
  `quant/gguf/*.gguf`. llama.cpp built from a pinned commit in `third_party/llama.cpp` (CUDA
  build for GPU bench, CPU build for CPU bench); commit hash recorded.
Each export: size on disk, wall time, peak VRAM/RAM, tool versions. Failures are results.

## bench
- `bench/runner.py`: one function per backend — vLLM (bf16/FP8/AWQ/GPTQ) and llama.cpp
  (`llama-server` with `--n-gpu-layers 999` for CUDA, `0` for CPU; OpenAI-compatible HTTP; temp 0;
  JSON grammar off to match the primary eval) — producing the same `predictions/<set>.jsonl`
  format as `train/predict.py`, plus per-unit latency and token counts. Then `regslm evaluate`.
- Throughput/latency protocol: warm-up 20 units; then (a) offline throughput on 500 test units
  with max concurrency the backend allows; (b) single-stream latency p50/p95 on 100 units.
  VRAM via the 1 Hz `gpu.csv` trace; host RAM via `/proc/<pid>/status` VmHWM for llama.cpp CPU.
- **Speculative decoding**: vLLM with 0.6B SFT draft → 8B SFT target (`speculative_config`,
  `num_speculative_tokens` ∈ {3, 5}); report acceptance rate, throughput gain, and verify outputs
  are identical to the target alone at temperature 0 (count mismatches; mismatches are an anomaly).
- **Needle-in-rulebook**: synthetic contexts of 4k/8k/16k tokens made of consecutive real units
  from one held-out document with a single target unit (with a known silver obligation)
  inserted at depth 10/50/90 %; task = extract obligations for unit_id X only; score = OF1 on the
  target + count of obligations attributed to other units. Run only where the context fits.
- **CPU portability**: llama.cpp CPU-only on the i5-13500 (threads = 6 P-cores, and 14 all
  cores); `scripts/bench_mac.sh` for an Apple M4 (owner runs it; writes JSON we ingest).
- Results: `results/bench/<run>/<backend>-<format>.json`; aggregated by
  `experiments/rq4_quant_table.py`.

Tests (CPU): llama.cpp HTTP client against a stub server; latency/throughput aggregation;
needle-context builder (exact token lengths with a fake tokenizer, depth placement).
