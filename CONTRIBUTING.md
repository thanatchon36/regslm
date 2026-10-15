# Contributing

regslm is an independent open-source project built on **public documents only**. Read
[DISCLAIMER.md](DISCLAIMER.md) first. Never contribute non-public material or anything that
reflects an institution's internal practice.

## Dev workflow

- Python 3.11+, [uv](https://docs.astral.sh/uv/). `make setup` builds `.venv` (core + vLLM);
  `make setup-train` builds `.venv-train` (Unsloth/TRL); `make setup-quant` builds `.venv-quant`.
  vLLM and Unsloth pin different torch builds (docs/DECISIONS.md D-002), so the envs are separate
  and Make targets use `uv run --no-sync`: never re-sync one env from another.
- Contributors without a GPU need only: `uv sync --extra th --extra pdf`, then `make test lint`
  (pytest with `not network and not gpu`, ruff, `mypy --strict` on core). CI runs exactly this.
- Core coverage target is >= 85 %. Tests that need a live endpoint are `@pytest.mark.network`;
  GPU tests are `@pytest.mark.gpu`. Guard heavy imports with `pytest.importorskip`.
- Long jobs (train, label, predict, quantize, bench) always run through `scripts/tmux_job.sh`
  (the Make targets do this): detached tmux, 1 Hz `gpu.csv`, `exit.json`, `DONE` marker. Never run
  a training process in a foreground terminal. Chain GPU jobs with `scripts/run_queue.sh`.
  Peak VRAM must stay <= 11 GB; an OOM is recorded as a result, not silently reconfigured.
- Record every design decision in `docs/DECISIONS.md` (next free `D-nnn`).

## Add a regulator source

1. Read `docs/specs/_adapter_common.md` (the adapter contract) and `src/regslm/corpus/base.py`.
2. Add `src/regslm/corpus/<name>.py` with a `SourceAdapter` subclass; the module docstring states
   document types, URL patterns, structure, language and licence terms with a URL.
3. Fetch only through `self.client` (robots.txt, <= 1 request / 2 s, descriptive User-Agent, disk
   cache). If a site blocks automation, do not evade it: list the URLs in `docs/sources/NEEDED.md`.
4. Register it in `src/regslm/corpus/registry.py`.
5. Add small real fixtures in `tests/fixtures/<name>/` (<= 200 KB), offline tests in
   `tests/test_corpus_<name>.py` and one `@pytest.mark.network` smoke test.
6. Record the terms in `docs/sources/LICENSES.md` **before committing any text**. Open/public-domain
   sources may have text committed; unclear or restrictive sources commit only URLs, hashes and the
   fetch script, and the text stays local. Weights are released only if trained on text whose terms
   allow derivatives.

## Add a language

- Segmentation: sentence/paragraph unit splitting is per adapter; word segmentation for scripts
  without spaces uses optional extras (Thai: `pythainlp`, extra `th`). Add the extra to `pyproject.toml`
  if you need one, and keep it out of the core dependencies.
- Triggers: extend the deontic `TRIGGER` regex in `src/regslm/corpus/sample.py` with the language's
  obligation/prohibition/permission markers, and add test cases to `tests/test_sample.py`.
- Guide rules: describe language-specific ambiguities and decisions in `docs/ANNOTATION_GUIDE.md`
  (and `docs/INTERPRETATIONS.md`); bump the guide/prompt version in `src/regslm/teacher/prompts.py`.
- Splits: set the language handling in `config/splits.yaml`. Report per-language results; do not
  generalise from one language.

## Add a reward function

- Spec: `docs/specs/rewards.md`. Components live in `src/regslm/rewards/components.py` as
  `fn(completion, unit, reference) -> float in [0, 1]` and are combined in `rewards/composite.py`
  with weights from `config/rewards.yaml`.
- Rewards must be verifiable (computed from the unit text and/or schema, no model judge), cheap,
  and hard to game: add a test that a degenerate completion (empty, copied text, hallucinated number)
  scores low. `mypy --strict` applies to `src/regslm/rewards`.
- Changing the reward set after pre-registration is a deviation: log it in `docs/DECISIONS.md`.

## Add a base model

1. Add an entry under `bases:` in `config/grid.yaml` (HF id, methods, and an assumed throughput in
   `assumed_train_tok_s` / `assumed_infer_units_h`).
2. Pin the exact HF revision in `results/base_revisions.json` (the grid refuses unpinned bases).
3. Check the licence (gated or non-Apache bases must be named and licensed accordingly in model
   cards) and record availability in `docs/DECISIONS.md`.
4. Re-project the budget with `experiments/budget.py` (GPU-hours only, never metrics). Total cap
   100 GPU-h; apply the pre-registered cut order in `docs/RESEARCH_QUESTIONS.md` section 7 if it is exceeded.
5. Verify the 11 GB VRAM ceiling with `make quick`-style smoke before adding it to a priority block.
