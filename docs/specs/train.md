# Spec: `regslm.train` — SFT / distillation / run registry (paper §4.1–4.3)

Environment: `.venv-train` (`UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync ...`), Unsloth
+ TRL + PEFT + bitsandbytes. Fallback to plain TRL/PEFT when Unsloth lacks a base (record which
path was used in `metrics.json` as `trainer_backend`).

## Data builder (`train/data.py`)
- Inputs: teacher `labels.jsonl` files (resolved `Extraction`s), `data/splits/assignments.parquet`,
  units parquet. Only split == train. Call `eval.gold.assert_no_gold(unit_ids)` (or an equivalent
  local check against `data/gold/sample.jsonl` if that module does not exist yet).
- Target = `GenExtraction` JSON rebuilt from the resolved label: `citation.quote` =
  `text[span.start:span.end]` (D-020). Label-only arm: `rationale: null`. Rationale arm: teacher's
  rationale text kept (rationale-variant labels). Compact JSON (no indentation), keys in schema
  order.
- Drop units whose label was invalid or had unresolved obligations (report counts).
- Rule-empty units (no deontic trigger, D-027): added at the train-split ratio
  (#untriggered/#triggered among train units) × N, target `{"unit_id":…, "rationale":null,
  "obligations":[]}`; `origin` column records `teacher` vs `rule_empty`.
- `N` = number of teacher-labelled units; subsets are **nested** (N=500 ⊂ 2k ⊂ 10k ⊂ 20k) by a
  fixed shuffle (seed 0, independent of the training seed) so label-efficiency curves are
  comparable. Equal-token-budget RQ2 arm: label-only gets units added (in the same nested
  order) until its total target tokens match the rationale arm's (tokenizer of the base model).
- Prompt for students (`train/prompts.py`): a **short** system prompt (≤ 120 tokens: task,
  schema field list, "quote the trigger sentence from TEXT", "empty list if no obligation") —
  the long teacher rules are not given to students; user prompt = `teacher.prompts.user_prompt`.
  Same prompts for every base; chat template of each base; Qwen3 `enable_thinking=False`.
- Sequences > 2,048 tokens (prompt + target) are dropped and counted (D-029).

## Trainer (`train/sft.py`)
Config (pydantic, YAML in `results/runs/<id>/config.yaml`):
`base, revision, method ∈ {lora, qlora, full}, n_labels, arm ∈ {label, rationale, placebo},
seed, lr, epochs, min_steps, per_device_batch, grad_accum, lora_r, lora_alpha, lora_dropout,
target_modules, max_seq_len=2048, optim, bf16=True, grad_ckpt=True, eval_n`.
Defaults (pre-registered): LoRA r=16, α=32, dropout 0, all linear projections; lr 2e-4 (LoRA/
QLoRA), 2e-5 (full; 8-bit AdamW for 1.7B); cosine, warmup 3 %; effective batch 16; epochs = 2,
but at least `min_steps=300` optimizer steps (small N get more epochs); loss on completion tokens
only; packing off. Placebo arm: targets permuted across units (seeded) before training.
Determinism: `torch.use_deterministic_algorithms(True, warn_only=True)`, seeds everywhere;
record kernels that warned in `metrics.json:nondeterministic_ops`.

Outputs per run (`results/runs/<id>/`): `config.yaml`, `train.log` (via tmux wrapper),
`gpu.csv` (wrapper), `metrics.json` (train loss curve summary, tokens seen, wall time, peak
VRAM from torch, base revision sha, tokenizer sha256 of tokenizer.json, config hash,
trainer_backend), adapter/checkpoint under `results/runs/<id>/checkpoints/` (git-ignored),
`DONE` (wrapper). OOM: catch `torch.cuda.OutOfMemoryError`, write
`metrics.json:{"status": "does_not_fit", ...}` and exit 0 (it is a result, MISSION §0) — the
wrapper then writes DONE.

## Inference for evaluation (`train/predict.py`, runs in `.venv` with vLLM)
- Merge LoRA into the base (bf16) or load full checkpoint → vLLM, temperature 0, max_tokens 1024,
  **no guided decoding** (validity is a measured property; guided decoding is an RQ4 serving
  variant). Outputs `predictions/<test_set>.jsonl` (raw completion + parsed/resolved Extraction +
  n_unresolved + latency). Test sets: test, test_perturbed, heldout, postcutoff (subset of test
  flagged), silver_val, gold (if finalised), Thai test.
- Then `eval.metrics` → `results/runs/<id>/eval.json` (all metrics + bootstrap CIs).

## CLI / Make
`regslm train --run-id ID` (reads config.yaml in run dir; `regslm train-config` writes one from
flags/grid), `regslm predict --run-id ID --sets test,test_perturbed,...`, `regslm evaluate
--run-id ID`. `make train RUN=..`, `make quick`: builds a 200-example fixture (from silver labels
if present, else from `tests/fixtures/quick_labels.jsonl`), trains Qwen3-0.6B LoRA for 50 steps,
predicts on 50 units, evaluates — all ≤ 15 min, wall time logged to `results/timing.json`.

## Tests (CPU, no GPU)
Data builder (nested subsets, rule-empty ratio, gold exclusion assertion, quote reconstruction,
placebo permutation, token-budget matching with a fake tokenizer), config hashing, OOM path via a
monkeypatched trainer.
