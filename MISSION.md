# MISSION: regslm — How Small Can a Regulatory-Obligation Extractor Be? A Scaling × Method × Quantization Study with a Fully Offline Pipeline

You are an autonomous research engineer. Build **`regslm`**, a public, pip-installable Python library that turns public financial rulebooks into structured **obligations** (who must do what, under which condition, with which threshold/deadline, citing which paragraph) using small language models that run entirely on-premises — and run a **pre-registered empirical study** of how model size, adaptation method (SFT, rationale distillation, GRPO with verifiable rewards), label budget and quantization trade off against each other, with **model weights released** on Hugging Face.

Everything — teacher labelling, training, evaluation, serving — runs on one consumer GPU with no hosted API in the loop. That constraint is the point: supervisory and compliance data cannot leave the institution, so the question a regulator or bank actually asks is "what is the smallest model I can run in my own rack that does this job?" This project answers it with numbers.

Half library, half research. The library must be usable tomorrow: `regslm fetch pra fca ecfr && regslm extract --model regslm/qwen3-1.7b-oblig-awq doc.pdf`. The paper is evidence that the library works.

Work through all phases without waiting for me. Record every design decision in `docs/DECISIONS.md`. Only stop if truly blocked.

---

## 0. Hard constraints

- **Public documents only.** Rulebooks, handbooks, regulations and guidelines from regulators' public websites or official legal portals. No non-public material, no internal documents, no knowledge of any institution's supervisory practice. Every text in the corpus traces to a URL + SHA-256 in `data/corpus/manifest.parquet`.
- **Author context.** The author works at a central bank. README, `DISCLAIMER.md`, model cards and paper state: independent open-source project; public documents only; no institution's view; not legal, regulatory or compliance advice; extraction output must be reviewed by a qualified person before any use. Thai legal texts are included as *one language among several* with no commentary on any Thai institution's policy.
- **Licensing of text.** Record terms per source in `docs/sources/LICENSES.md` before committing any text. Public-domain or open-licence sources (US eCFR, EUR-Lex with attribution, BIS Basel Framework with attribution, Thai statutes — verify whether the Thai Copyright Act excludes laws/regulations from protection, and cite the section) may be committed. Where terms are unclear or restrictive (PRA Rulebook, FCA Handbook, HKMA SPM, MAS), commit only URLs + hashes + a fetch script and keep text local; train on it only if its terms allow research use, and release model weights trained only on text whose terms allow derivative use — otherwise train two variants and release the permissive one. Honour `robots.txt`, rate-limit ≤ 1 req / 2 s, descriptive User-Agent, cache raw responses. If a site blocks automation, write `docs/sources/NEEDED.md` with exact URLs for me and continue.
- **Fully offline pipeline.** The teacher is a local model (default **Qwen3-14B, 4-bit AWQ or GPTQ, served by vLLM**). A hosted-API teacher is an optional extra (`[api]`) used for at most one sanity comparison on ≤ 500 items and never required by `make reproduce`.
- **Compute target: desktop, Ubuntu on WSL2 (Windows 11), Intel i5-13500 (6P+8E), RTX 4070 Ti 12 GB, 32 GB RAM.** Design everything for it:
  - VRAM is the binding constraint. **Peak VRAM ≤ 11 GB** on every run; log `nvidia-smi --query-gpu=memory.used --format=csv` at 1 Hz to `results/runs/<id>/gpu.csv`. Any run that OOMs is recorded as a result ("does not fit"), not silently reconfigured.
  - Model ladder: **Qwen3-0.6B (full FT + LoRA), Qwen3-1.7B (full FT with 8-bit AdamW + LoRA), Qwen3-4B (LoRA bf16 + QLoRA), Qwen3-8B (QLoRA only)**; second family **Gemma 3 1B / 4B** for the two smallest tiers; Thai-capable bases **Typhoon 2 and SEA-LION v3** at the ≤ 4B tier. Record exact HF revision for every base.
  - Training stack: Unsloth + TRL + PEFT + bitsandbytes, flash-attention 2, bf16, gradient checkpointing, sequence length 2,048 (4,096 only where a documented experiment needs it). GRPO via Unsloth/TRL. Fallback to plain PEFT/TRL if Unsloth lacks a base.
  - Serving/benchmark stack: vLLM (bf16 / FP8 / AWQ / GPTQ), llama.cpp with CUDA and CPU-only (GGUF Q4_K_M / Q8_0). FP8 is an Ada (SM 8.9) feature — verify the installed vLLM version supports it on this GPU before relying on it.
  - RAM: WSL2 must have `memory=24GB` and `swap=16GB` in `C:\Users\<me>\.wslconfig`; verify with `free -g` at startup and write the result to `results/hardware.json`. Keep the repo, `uv` cache and HF cache on the Linux filesystem (`~/`), never under `/mnt/c/` (10× slower I/O).
  - **Budgets:** full experiment suite ≤ **100 GPU-hours**, fully resumable (every run writes a completed marker; `make experiments` skips completed runs); `make quick` ≤ 15 min end-to-end on a 200-example fixture with the 0.6B model; teacher labelling ≤ 12 GPU-hours. Log per-stage wall time to `results/timing.json`.
  - Long jobs run under `tmux` with `nohup`-safe logging to `results/runs/<id>/train.log`; never a foreground training process that dies with the terminal. I will set Windows to never sleep; you cannot control that from WSL — remind me once in PROGRESS.md.
  - Reproducibility across CUDA non-determinism: **3 seeds per cell**, report mean ± 95 % CI; set `torch.use_deterministic_algorithms(True)` where it does not break kernels, and document where it does. Ship a `Dockerfile` (NVIDIA Container Toolkit on WSL2) pinning CUDA, torch, vLLM, Unsloth digests; `make reproduce-docker` is the canonical path.
- **Pre-registration.** `docs/RESEARCH_QUESTIONS.md` with hypotheses, metrics, thresholds, minimum cell sizes — committed and tagged `prereg-v1` before any training run beyond `make quick`.
- **Contamination is a first-class threat.** Public rulebooks are in every base model's pretraining data, so a zero-shot baseline may be reciting, not reading. Mandatory controls: (a) a **numerically-perturbed test set** — thresholds, percentages, deadlines and dates in the source text are changed; a model that outputs the original value instead of the perturbed one is reading from memory, and the *memorisation rate* is reported per model; (b) a **held-out rulebook** never used in training; (c) a **post-cutoff slice** of rules amended after each base model's documented training cutoff. A claim about extraction ability is made only on (a)-corrected numbers.
- **Every headline number has a sanity test.** A metric at exactly 0 / 1 / 100 %, a per-field F1 above 0.98, or a quantized model scoring higher than its bf16 parent by more than the CI width fails the build until explained in `docs/ANOMALIES.md`.
- **Honest science.** Refuted hypotheses are reported as refuted. Where a smaller or cheaper method wins, say so plainly.
- Python 3.11+, `uv`, `pydantic` v2, `typer`, `polars`, `pytest` (≥ 85 % on core), `ruff`, `mypy --strict` on core, type hints. Apache-2.0 for code and for released weights where the training data permits; corpus under per-source terms.

---

## 1. Phase 1 — Corpus, obligation schema, gold and silver labels, literature

### 1.1 Sources (`src/regslm/corpus/`)
One adapter per source with docstring (document types, URL patterns, structure, language, terms):

| Source | Language | Notes |
|---|---|---|
| US eCFR Title 12 (banking) + Title 17 (securities), selected parts | en | public domain; clean XML API |
| EUR-Lex: CRR (575/2013), CRD, MiFID II, AMLD — consolidated versions | en (+ other EU languages available) | reuse with attribution |
| BIS Basel Framework (consolidated) | en | attribution required |
| UK PRA Rulebook; FCA Handbook (selected sourcebooks: SYSC, COBS, SUP) | en | check terms; likely local-only text |
| HKMA Supervisory Policy Manual | en | check terms |
| MAS Notices and Guidelines (banking) | en | site blocks bots → NEEDED.md path |
| Thai: Financial Institutions Business Act, SEC Act, selected BoT / SEC notifications published in the Royal Gazette | **th** (+ official en translations where they exist, as a parallel set) | verify copyright status of statutes; treat Thailand as one language among several |

Parse to clean, hierarchically-addressed units (`source / part / section / paragraph`) so every obligation carries a resolvable citation. Thai segmentation via `pythainlp`. Build `documents.parquet`, `units.parquet`, `manifest.parquet`, versioned under `data/corpus/VERSION`.

### 1.2 Obligation schema (`src/regslm/schema/`)
Pydantic model + JSON Schema:
`Obligation(subject, modality ∈ {must, must_not, may, should}, action, object, condition, threshold{value, unit, comparator}, deadline, exceptions[], citation{source, unit_id, span}, confidence)`.
Write `docs/ANNOTATION_GUIDE.md` with decision rules and ≥ 30 worked examples including hard cases (nested conditions, cross-references, definitions that are not obligations, "may" vs "must").

### 1.3 Gold set
- **300 units** stratified across sources and languages, annotated by **two independent local-LLM passes with different prompts (14B teacher and 8B model) → disagreements adjudicated by me**. Produce `data/gold/to_adjudicate.jsonl` with both candidate annotations side by side and a one-line instruction; put it under "Waiting on owner" in PROGRESS.md and **continue** with silver labels. Until I adjudicate, report results on the silver-validated split and mark gold-set numbers as pending. Report inter-pass agreement (span F1, field-level κ).
- Gold is **never** used for training or prompt tuning.

### 1.4 Silver labels (teacher)
- Teacher: local Qwen3-14B 4-bit via vLLM, structured JSON output, two variants: **label-only** and **label + short rationale**. Cache by content hash; ≤ 12 GPU-hours total; target 30–50k labelled units.
- Teacher quality is measured on gold before any student is trained; the teacher's own numbers are a row in every results table.

### 1.5 Splits
- `train / dev / test` by unit with **no unit from a held-out rulebook** (default held-out: HKMA SPM or FCA SYSC — choose the one with permissive terms) in train; **perturbed test** (numeric perturbation of `test`); **post-cutoff test** (units amended after the base models' cutoffs; may be small — report N).

### 1.6 Literature (`docs/literature/REVIEW.md`, `refs.bib`, 25–35 sources)
Obligation / deontic extraction and regulatory NLP (LegalBench, CUAD, ObliQA, regulatory-text IE work), knowledge distillation and rationale distillation for small models (Distilling step-by-step, Orca-style), RL with verifiable rewards (DeepSeek-R1, GRPO, RLVR on structured tasks), label-efficiency and small-model scaling, quantization effects beyond perplexity (AWQ, GPTQ, FP8, GGUF studies), contamination/memorisation of public text, Thai legal NLP and tokenizer efficiency, SupTech/RegTech applications of LLMs (BIS/MAS Ellipse notes). End with a gap table.

---

## 2. Phase 2 — Research questions (pre-registered, tag `prereg-v1`)

- **RQ1 — Crossover / label efficiency.** For each size, how does extraction F1 (span + field) scale with label budget N ∈ {500, 2k, 10k, 50k}? At what N does a fine-tuned 1.7B beat the 8B zero-shot and the 14B teacher? Pre-set: a crossover is "found" only if the CI of the difference excludes zero at that N and the next.
- **RQ2 — Rationale distillation vs label-only.** At equal token budget, does training on teacher rationales improve student F1, especially on `condition` and `exceptions`? Does the gain shrink with model size?
- **RQ3 — GRPO with verifiable rewards.** Starting from the best SFT checkpoint per size (0.6B–4B), does GRPO with a programmatic reward — schema validity + citation resolvable + span grounded in source + threshold/deadline exact match — improve precision on numeric fields and reduce hallucinated obligations? Report reward hacking checks (e.g., degenerate short outputs) explicitly.
- **RQ4 — Quantization and serving.** Per size: bf16 vs FP8 vs AWQ-4 vs GPTQ-4 vs GGUF Q4_K_M/Q8 — where is the F1 cliff, and does it hit numeric fields first? Throughput, latency and VRAM under vLLM and llama.cpp; speculative decoding (0.6B draft → 8B target) net gain; needle-in-rulebook at 4k/8k/16k context; **CPU-only (i5-13500) and Apple M4 portability** (M4 run is a script I execute and whose JSON you ingest; optional).
- **RQ5 — Thai vs English.** Same schema, same methods: F1 gap, tokenizer cost (tokens per obligation, latency per document), whether Thai-pretrained bases (Typhoon, SEA-LION) close the gap at 1–4B, and whether English fine-tuning transfers zero-shot to Thai.
- **Mandatory cross-cutting:** memorisation rate on the perturbed test set per model and method; held-out-rulebook generalisation gap; placebo with shuffled labels.

Each RQ: hypothesis, refuting experiment, metric, threshold, minimum N, what is reported if underpowered.

---

## 3. Phase 3 — Library (`src/regslm/`)

- **`corpus`** — adapters, parsers (XML/HTML/PDF → units), segmenters (en/th), manifest/snapshot.
- **`schema`** — `Obligation`, validators, JSON Schema export, perturbation utilities (`perturb_numbers(unit) -> (unit', mapping)`).
- **`teacher`** — vLLM-backed structured labelling with content-hash cache, rationale/no-rationale prompts, batch runner.
- **`train`** — `sft` (LoRA / full, Unsloth → TRL fallback), `distill` (rationale-augmented SFT), `grpo` (reward composition, KL, sampling config), unified run config (YAML) + run registry (`results/runs/<id>/{config.yaml, metrics.json, gpu.csv, train.log, DONE}`).
- **`rewards`** — pure functions: `schema_valid`, `citation_resolves`, `span_grounded`, `numeric_exact`, `deadline_exact`, `no_hallucinated_obligation`; each unit-tested; composite weights in config.
- **`quant`** — export to AWQ / GPTQ / FP8 / GGUF with recorded tool versions.
- **`serve`** — vLLM and llama.cpp runners; `regslm extract` CLI on a PDF/HTML/text file → JSONL obligations with citations.
- **`bench`** — F1 (span, field, exact), memorisation rate, latency/throughput/VRAM, needle tests, CPU/Mac portability scripts.
- **`eval`** — bootstrap CIs, paired tests across seeds, crossover detection, Holm correction across sizes.
- **CLI:** `regslm fetch | parse | snapshot | label | split | train | grpo | quantize | bench | extract | report | doctor` (`doctor` checks CUDA, VRAM, WSL memory, disk, vLLM FP8 support).
- JSON Schemas under `schemas/`; every public function cites the paper section it implements.

---

## 4. Phase 4 — Experiments (`experiments/`)

- One script per figure/table; grid in `config/grid.yaml`; cells = size × family × method × N × seed. Prioritised ordering so the most informative cells complete first: (1) zero-shot + teacher on all test sets, (2) SFT-LoRA ladder at N = 10k, (3) label-efficiency curves, (4) distillation, (5) GRPO on best SFT, (6) quantization sweep on best checkpoint per size, (7) Thai.
- Enforce budgets: stop and record if the projected GPU-hours exceed the cap; shrink seeds for the largest size first and say so.
- Every run: VRAM trace, wall time, exact base revision, tokenizer hash, config hash. Placebo and perturbed-set results sit next to every real result.

---

## 5. Phase 5 — Model release, paper, communications

- **Hugging Face release** (only weights trained on permissively-licensed text): best checkpoint per size in bf16 + AWQ + GGUF, with model cards (intended use, data, licences, eval tables incl. memorisation rate, limitations, "review before use"), and a dataset card for the redistributable corpus + silver labels. Org/user: mine; names `regslm-qwen3-<size>-oblig[-awq|-gguf]`.
- `paper/paper.md` → PDF via pandoc. Working title: *How Small Can a Regulatory-Obligation Extractor Be? Scaling, Distillation, Verifiable-Reward RL and Quantization for On-Premises SupTech.* Sections: Abstract, Introduction, Related Work, Data & Schema, Methods, Experimental Setup (hardware, budgets, pre-registration), Results per RQ, Memorisation & Contamination, Threats to Validity (silver labels, gold pending, single GPU, English-heavy corpus), Reproducibility, Conclusion. Numbers injected from `results/summary.json`.
- `docs/EXEC_SUMMARY.md` — one page for a head of supervision / CRO: what size runs on what hardware for what accuracy, what still needs a human, what it costs.
- `docs/posts/` — 8–10 LinkedIn drafts, one finding each, no hype, repo + HF links.

---

## 6. Phase 6 — Packaging and publication

```
regslm/
├── README.md · MISSION.md · DISCLAIMER.md · LICENSE (Apache-2.0)
├── pyproject.toml        # uv; extras: [train] [serve] [gguf] [api] [docs]
├── Dockerfile · docker-compose.yml   # NVIDIA Container Toolkit; make reproduce-docker
├── Makefile              # setup | doctor | quick | fetch | snapshot | label | train | grpo | quantize | bench | experiments | paper | reproduce | reproduce-docker | test | lint | docs | release
├── config/ (grid.yaml, paper.yaml, rewards.yaml)
├── src/regslm/ · schemas/ · data/ (raw git-ignored; corpus manifest + permissive text committed) · experiments/ · results/ · paper/ · docs/ · tests/
└── .github/workflows/ (ci.yml: lint + tests + quick on a CPU fixture, no GPU in CI; release.yml)
```

- PyPI release on `v0.1.0` after `make reproduce-docker` passes from a fresh clone; `CITATION.cff`; Zenodo DOI; GitHub topics (`suptech`, `regtech`, `small-language-models`, `fine-tuning`, `grpo`, `quantization`, `thai-nlp`).
- `CONTRIBUTING.md`: adding a regulator source, a language, a reward function, a base model.

---

## 7. Working rules

- Plan each phase in `docs/PLAN.md`; close in `docs/PROGRESS.md` (shipped / cut / open / **Waiting on owner**). Keep `docs/FUTURE_WORK.md`, `docs/ANOMALIES.md`, `docs/INTERPRETATIONS.md` (annotation-guide ambiguities).
- **Token budget (strict):** the main thread does design, schema, annotation guide, pre-registration, reward design, interpretation and writing. Delegate to Sonnet subagents with written specs (`docs/specs/<module>.md`) and acceptance tests: each source adapter, parsers, training wrappers, quantization export, bench harness, tests, docs site, CI, Dockerfile. One subagent per adapter; one per literature cluster. Review diffs; do not redo work. Never load whole rulebooks/PDFs into context — parse to files, read units on demand. Never paste training logs into context — grep them.
- **GPU etiquette:** one training job at a time; check `nvidia-smi` before launching; launch via `make train RUN=<id>` under tmux; poll `results/runs/<id>/metrics.json`, not stdout. If a run OOMs, record it as a result and move on.
- **Handoff protocol:** update `docs/PROGRESS.md` at every phase end and whenever context passes ~60 % (remaining checklist, undocumented decisions, running jobs and how to check them); commit; push; a fresh session resumes with "Read MISSION.md, docs/PROGRESS.md, docs/DECISIONS.md and continue".
- Small tested commits; `make test lint` before every commit; push after each phase.
- When choosing between "one more base model family" and "clean curves on the ladder we have", choose clean curves. When choosing between "a higher F1 from a bigger model" and "a result that survives the perturbed test set", choose the latter.

Start now: run `regslm doctor` equivalent checks (CUDA, VRAM, WSL memory, disk, vLLM FP8), write `results/hardware.json`, then Phase 1 beginning with the eCFR and EUR-Lex adapters (cleanest sources, fastest to validate the pipeline).
