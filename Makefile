# All long-running jobs go through scripts/tmux_job.sh (detached tmux, GPU trace, DONE marker).
SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)
UV := uv run --no-sync
RUN ?=
JOB := scripts/tmux_job.sh

.PHONY: setup setup-train doctor test lint fmt schema fetch parse snapshot teacher-smoke label quick train experiments

setup:            ## core + serving env (.venv)
	uv sync --extra serve --extra th --extra pdf
setup-train:      ## training env (.venv-train; torch pinned by Unsloth)
	UV_PROJECT_ENVIRONMENT=.venv-train uv sync --extra train --extra th
doctor:
	$(UV) regslm doctor > /dev/null && cat results/hardware.json | head -40
test:
	$(UV) pytest -q --cov=regslm --cov-report=term-missing:skip-covered
lint:
	$(UV) ruff check src tests && $(UV) mypy
fmt:
	$(UV) ruff format src tests && $(UV) ruff check --fix src tests
schema:
	$(UV) regslm schema
fetch:            ## SOURCES="ecfr eurlex"
	$(UV) regslm fetch $(SOURCES)
teacher-smoke:    ## load Qwen3-14B-AWQ in vLLM and label 8 units; measures VRAM headroom
	$(JOB) teacher-smoke $(UV) python scripts/teacher_smoke.py
label:            ## RUN=label-<variant> ARGS="--units X.jsonl --variant label [--limit N]"
	$(JOB) $(RUN) $(UV) regslm label --run-id $(RUN) $(ARGS)
train:            ## RUN=<id> (config in results/runs/<id>/config.yaml)
	$(JOB) $(RUN) env UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync regslm train --run-id $(RUN)
