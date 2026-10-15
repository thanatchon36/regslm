# All long-running jobs go through scripts/tmux_job.sh (detached tmux, GPU trace, DONE marker).
SHELL := /bin/bash
export PATH := $(HOME)/.local/bin:$(PATH)
UV := uv run --no-sync
RUN ?=
JOB := scripts/tmux_job.sh

.PHONY: quantize bench grpo grpo-config predict evaluate bench-cell train-config setup setup-train doctor test lint fmt schema fetch parse snapshot teacher-smoke label quick experiments

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
train-config:     ## RUN=<id> ARGS="--base Qwen/Qwen3-1.7B --n-labels 2000 --set lr=1e-4"
	env UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync regslm train-config --run-id $(RUN) $(ARGS)
grpo:             ## RUN=<id> (config.yaml + grpo.yaml in results/runs/<id>/)
	$(JOB) $(RUN) env UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync regslm grpo --run-id $(RUN)
grpo-config:      ## RUN=<id> ARGS="--parent <sft-run> | --auto-base Qwen/Qwen3-1.7B [--set beta=0.02]"
	env UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync regslm grpo-config --run-id $(RUN) $(ARGS)
predict:          ## RUN=<id> SETS=test,test_perturbed or ARGS="--sets ..."  (vLLM env; marker dir results/runs/<id>/predict/)
	$(JOB) $(RUN)/predict $(UV) regslm predict --run-id $(RUN) $(if $(SETS),--sets $(SETS)) $(ARGS)
evaluate:         ## RUN=<id> [ARGS="--sets test"]  (CPU; under tmux for exit.json/DONE in results/runs/<id>/evaluate/)
	$(JOB) $(RUN)/evaluate $(UV) regslm evaluate --run-id $(RUN) $(ARGS)
quick:            ## 200-row fixture -> Qwen3-0.6B LoRA 50 steps -> predict 50 -> evaluate (<= 15 min)
	rm -rf results/runs/quick
	$(JOB) quick bash scripts/quick.sh
QUANT_FORMATS = fp8,awq,gptq,gguf-q8,gguf-q4km
quantize:         ## RUN=<id> [FORMATS=fp8,awq,gptq,gguf-q8,gguf-q4km | ARGS=<same>]  (.venv-quant; needs GPU for AWQ/GPTQ)
	$(JOB) $(RUN)-quantize env UV_PROJECT_ENVIRONMENT=.venv-quant uv run --no-sync regslm quantize --run-id $(RUN) --formats $(or $(FORMATS),$(ARGS),$(QUANT_FORMATS))
bench:            ## RUN=<id> BACKEND=vllm|llamacpp-cuda|llamacpp-cpu FORMAT=bf16|fp8|awq|gptq|gguf-q8|gguf-q4km [ARGS=--needle]
	$(JOB) $(RUN)-bench-$(BACKEND)-$(FORMAT) $(UV) regslm bench --run-id $(RUN) --backend $(BACKEND) --format $(FORMAT) $(ARGS)

bench-cell:       ## RUN=q-<base>-<fmt>-<backend> ARGS="<parent-run> <backend> <format>"  (experiments/run.py quant cells)
	$(JOB) $(RUN) $(UV) regslm bench --run-id $(word 1,$(ARGS)) --backend $(word 2,$(ARGS)) --format $(word 3,$(ARGS))

# ---- Phase 6: end-to-end reproduction (may take days; resumable, every stage skips DONE work) ----
.PHONY: setup-quant parse snapshot split paper reproduce reproduce-docker queue-run
setup-quant:      ## quantization env (.venv-quant)
	UV_PROJECT_ENVIRONMENT=.venv-quant uv sync --extra quant --extra th
parse:            ## SOURCES="ecfr eurlex"
	$(UV) regslm parse $(SOURCES)
snapshot:
	$(UV) regslm snapshot
split:
	$(UV) regslm split
experiments:      ## [BLOCK=name] [DRY=1] -> results/queues/<block>.txt (experiments/run.py, docs/specs/experiments.md)
	$(UV) python -m experiments.run $(if $(BLOCK),--block $(BLOCK)) $(if $(DRY),--dry)
paper:            ## inject results/summary.json into paper/paper.md -> PDF via pandoc
	$(UV) python paper/build.py
queue-run:        ## QUEUE=<file>: run a queue file strictly sequentially (blocks until the last job ends)
	scripts/run_queue.sh $(QUEUE)
reproduce:        ## fetch -> parse -> snapshot -> split -> teacher labels -> perturbed test -> experiments -> paper
	$(MAKE) setup setup-train setup-quant doctor
	$(MAKE) fetch parse snapshot split
	$(MAKE) queue-run QUEUE=config/queue_phase1.txt
	$(UV) regslm perturb-test --labels results/runs/silver-eval-all-label/labels.jsonl
	$(MAKE) experiments
	for q in results/queues/*.txt; do $(MAKE) queue-run QUEUE=$$q || exit 1; done
	$(MAKE) paper
reproduce-docker: ## canonical path: needs NVIDIA Container Toolkit (docker compose run --gpus via compose file)
	docker compose run --rm regslm make reproduce
