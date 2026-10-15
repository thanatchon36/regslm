# Progress

## Waiting on owner
- **WSL memory/swap below target.** `free -g` shows 15 GB RAM / 4 GB swap; mission requires `memory=24GB` and `swap=16GB`. Add to `C:\Users\<you>\.wslconfig` under `[wsl2]` (keep `networkingMode=mirrored`), then `wsl --shutdown` from PowerShell. Until then: teacher/training run but full-FT 1.7B and GGUF conversion of 8B may hit host-RAM limits.
- **Windows sleep.** Set Windows power plan to never sleep while long jobs run (cannot be controlled from WSL).
- **FCA Handbook access route.** Text is OGL v3.0 (permissive) but FCA terms cl. 35.3 forbid scraping without consent. Either request FCA consent / API access, or download the SYSC, COBS and SUP sourcebook PDFs manually into `data/local/manual/fca/` (see docs/sources/NEEDED.md). FCA SYSC is the planned held-out rulebook (D-012).
- **pandoc** not installed (`sudo apt install pandoc texlive-xetex`) — only needed for the paper PDF (Phase 5).

## Phase 1 — in progress (2026-10-15)
### Shipped
- Hardware report (`results/hardware.json`): RTX 4070 Ti 12 GB (CC 8.9; Windows holds 0.5–1.5 GB), i5-13500, 15 GB RAM / 4 GB swap (below target), vLLM 0.31.0 + torch 2.13/cu130, FP8 supported.
- Skeleton, schema (+ JSON Schema export), numeric perturbation (EU/Thai formats), corpus contracts with robots-safe PoliteClient, tmux job runner (`scripts/tmux_job.sh`).
- Adapters: **eCFR** (GovInfo bulk XML; 9 docs, ~25.5k units), **EUR-Lex** (Cellar; CRR/CRD/MiFID2/AMLD4, ~12.2k units). Units in `data/local/units/*.jsonl` (until snapshot).
- Teacher: Qwen3-14B-AWQ fits (D-015); `regslm label` runner with SQLite content-hash cache; prompts v1 (label / rationale / gold_a / gold_b).
- Rewards module (100 % coverage); annotation guide v1 (36 examples); licence policy (tiers P/L/X); literature review (35 verified sources + gap table); pre-registration **draft** (`docs/RESEARCH_QUESTIONS.md`, not yet tagged).
### Running / in flight
- **`silver-eval-en-label`** (tmux, started 2026-10-15): teacher labels for test+dev+silver_val+heldout (8,457 units, ≈2.2 GPU-h). Poll `results/runs/silver-eval-en-label/metrics.json`; done when `DONE` exists. Resumable: re-launch the same make command — cached units are skipped.
- Subagent: Thai adapter (output so far in `data/local/units/thai*.jsonl`; not yet in the snapshot).
### Next
1. When eval labelling is done: `make label RUN=silver-train-en-label ARGS="--units data/local/jobs/train_trig_en.jsonl --variant label"` (19,918 units ≈ 5 h), then rationale variant on a 10k subset (D-027).
2. Thai into snapshot → gold sample (`corpus.sample.stratified_gold`) → re-run `regslm split` (gold excluded) → gold pass A (14B, prompt gold_a) + pass B (Qwen3-8B-AWQ, gold_b; downloaded) → `data/gold/to_adjudicate.jsonl`.
3. `regslm perturb-test` on test labels; inter-pass agreement metrics.
4. Finalise + tag `prereg-v1`; `make quick`; training stack (`.venv-train`).
### How to resume
`Read MISSION.md, docs/PROGRESS.md, docs/DECISIONS.md and continue`. Jobs: `tmux ls`; completion = `results/runs/<id>/DONE`; failure = `exit.json` with non-zero code.
