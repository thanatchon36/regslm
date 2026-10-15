# Spec: `regslm.teacher` — local vLLM labelling (paper §3.4)

Prompts are done (`src/regslm/teacher/prompts.py`, main-thread owned — do not edit).
Working engine settings are in docs/DECISIONS.md D-015 and `scripts/teacher_smoke.py`.

## Modules
- `teacher/engine.py` — `EngineConfig` (pydantic; model, revision, max_model_len=3072,
  gpu_memory_utilization=0.87, max_num_batched_tokens=1024, max_num_seqs=8,
  kv_cache_dtype="fp8", attention_backend="TRITON_ATTN", enforce_eager=False,
  enable_prefix_caching=True, temperature=0.0, max_tokens=1024, seed=0) and
  `build_llm(cfg)`; structured outputs from `Extraction.model_json_schema()` (handle both
  `StructuredOutputsParams` and older `GuidedDecodingParams` as in the smoke script);
  `chat_template_kwargs={"enable_thinking": False}`. vLLM imports are lazy (inside
  functions) so the module imports without vLLM installed.
- `teacher/cache.py` — content-addressed cache: key = sha256(model + revision +
  PROMPT_VERSION + variant + system prompt + user prompt + sampling params JSON). Storage:
  SQLite file `data/local/teacher_cache/<model-slug>.sqlite` (table: key PK, unit_id,
  variant, output TEXT, valid INT, n_prompt_tokens, n_gen_tokens, created_at). Must be safe to
  resume after kill -9 (commit per batch).
- `teacher/runner.py` — `label_units(units: Iterable[Unit], variant, cfg, run_dir, limit,
  batch_size=64)`: skip cached; batch through `llm.chat`; validate with
  `Extraction.model_validate_json`; on invalid JSON store raw output with valid=0 (no
  retries — validity rate is a reported metric); force `unit_id` to match the input; write
  `run_dir/labels.jsonl` (unit_id, variant, extraction or null, valid) at the end from the
  cache; write `run_dir/metrics.json` every batch: n_done, n_valid, valid_rate,
  gen_tokens, prompt_tokens, tok_per_s, units_per_hour, projected_hours_for(30000, 50000),
  elapsed_s. Periodic metrics are what the operator polls; keep stdout terse (one line per
  batch).
- CLI in `src/regslm/cli.py`: `regslm label --run-id ID --units PATH.jsonl --variant
  {label,rationale,gold_a,gold_b} [--model M] [--limit N] [--shuffle-seed S]`. Units input
  = JSONL of `corpus.base.Unit`. Run dir = `results/runs/<ID>/`; also dump the resolved
  `config.yaml` there. Add `label` target usage note to Makefile only if needed (a `label`
  target already exists calling `regslm label --run-id $(RUN)` — extend it to pass
  `ARGS="..."`).

## Tests (`tests/test_teacher.py`, no GPU)
Mock the LLM object (a fake with `.chat()` returning objects shaped like vLLM
`RequestOutput`): cache hit/miss, resume after partial run, invalid JSON recorded, unit_id
forced, metrics file shape, prompt variants produce different system prompts and label vs
rationale differ only by the rationale instruction. A `@pytest.mark.gpu` test that runs 4
units for real is allowed but not required.

Ruff clean; mypy not required for this module (vLLM types), but type hints everywhere.
