# Spec: experiment orchestration (`experiments/`, `make experiments`) — MISSION §4

## Cell expansion (`experiments/grid.py`)
- Read `config/grid.yaml` → list of `Cell` (pydantic): block, priority, base key, hf id + pinned
  revision (from `results/base_revisions.json`; refuse if missing), method ∈ {zero_shot, lora,
  qlora, full, grpo, quant}, n, arm ∈ {label, rationale, label_token_matched, placebo}, seed,
  train_langs, eval_sets, extra (max_wall_h, formats, …).
- `run_id` deterministic and readable: `{block}-{base}-{method}-n{n}-{arm}-{langs}-s{seed}`
  (zero-shot: `zs-{base}`; GRPO: `grpo-{base}-s{seed}`; quant: `q-{base}-{format}-{backend}`).
- Expansion rules match `experiments/budget.py` exactly (factor the shared logic out of
  budget.py into grid.py and make budget.py import it — the two must never disagree; test it).

## Runner (`experiments/run.py`, `make experiments [BLOCK=name] [DRY=1]`)
- For each cell in priority order: skip if `results/runs/<id>/DONE`; else write
  `config.yaml` via the existing `regslm train-config` / `grpo-config` / zero-shot predict config,
  then append the job line(s) to a queue file `results/queues/<block>.txt` in the format of
  `scripts/run_queue.sh` (train → predict → evaluate are separate queue lines for training cells:
  `train <id>`, `predict <id>`, `evaluate <id>`; add Makefile targets `predict`/`evaluate` taking
  `RUN=` if missing; predict/evaluate run under tmux too, so they get gpu.csv and DONE markers in
  sub-run dirs `results/runs/<id>/predict/`, `…/evaluate/` — or extend tmux_job.sh to accept a
  marker suffix; choose the simpler and document).
- Before emitting a block: run the budget projection (`grid.py` + measured throughput from
  `results/throughput.json` + logged GPU-hours so far from all `exit.json` / gpu.csv files);
  if logged + projected(block and all higher-priority remaining) > `budget_gpu_hours`, print the
  pre-registered cut that would be needed (RESEARCH_QUESTIONS §7 order) and **stop without
  emitting** (exit code 3). Never read metrics/eval files (test it: monkeypatch open to fail on
  eval.json/metrics.json paths inside budget code).
- `DRY=1` prints the cells, run ids and projected hours only.
- GPU-hour accounting (`experiments/ledger.py`): sum `wall_s` of every `exit.json` under
  `results/runs/` whose run is not a teacher-labelling run (teacher runs have ids starting
  `silver-`, `gold-pass-`, `teacher-`), write `results/gpu_ledger.json` {teacher_h, experiment_h,
  by_block}. Teacher time ≤ 12 h is reported separately.
- Throughput file (`experiments/throughput.py`): from finished training runs, compute
  tokens/s per base and method (`metrics.json:tokens_seen / train_wall_s` — note: these *are*
  metrics files, but only the throughput fields are read; keep this module separate from the
  budget decision code and document that it reads no quality metrics) and inference units/h from
  predict meta; write `results/throughput.json`.

## Training data materialisation
`regslm build-train-data --n N --arm A --langs en,th --seed-order 0` must already exist in
`train/data.py` (check); the runner calls it implicitly through train-config. Verify that nested
subsets are identical across bases (same unit ids for a given N).

## Zero-shot cells
Zero-shot = `regslm predict` with the base model (no checkpoint) and the student prompt, then
`regslm evaluate`. The 14B teacher zero-shot uses the student prompt too (not the teacher
prompt) — record this; its AWQ checkpoint runs through vLLM with the teacher engine settings.

## Tests
Expansion counts per block equal budget.py's; run-id uniqueness; skip-DONE; budget stop path
and that budget code never opens eval/metrics files; ledger sums; DRY output.
