# Progress

## Waiting on owner
- **WSL restart pending.** `.wslconfig` now has `memory=24GB` and `swap=16GB` (backup `.wslconfig.bak`). Run `wsl --shutdown` in PowerShell only after the GPU queue has finished (`tmux ls` shows no `queue`), then `free -g` and `make doctor`. Needed before full-FT 1.7B.
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
- **GPU queue** (tmux `queue`, `scripts/run_queue.sh config/queue_phase1.txt`, log `results/queue.log`), teacher prompt **v5**: silver-eval-all-label (9,203 units) → gold-pass-a (14B) → gold-pass-b (8B-AWQ) → silver-train-label (23,782) → silver-train-rationale (10k). ≈ 9.6 teacher GPU-h; ~1.3 used so far (pilots + stopped v3/v4 runs, D-036/D-038). If interrupted: restart the same tmux command; DONE runs are skipped and finished units are cached.
### Shipped since last update
- `make quick` passes end-to-end in 174 s (Unsloth train → merge → vLLM predict → evaluate).
- Guide v2.1 + prompt v5 after expert review; pre-registration v2 after methodology review (not yet tagged).
- Modules: eval (metrics/stats/agreement), gold build/finalize, train (SFT/predict/evaluate), GRPO, quant export (llmcompressor in `.venv-quant`), bench (llama.cpp b11630 CPU build + prebuilt CUDA 12.8 binaries), grid + budget projector (blocks 1–7 ≈ 101.5 GPU-h on assumed throughput).
### Next
1. When gold passes finish: `regslm gold build` → `data/gold/to_adjudicate.jsonl` → owner.
2. Measure real training throughput (first N=10k seed-0 run per size) → `results/throughput.json` → re-project budget → commit → tag `prereg-v1` before block 1.
3. Training data builder on silver labels; zero-shot block (inference only) can start right after labelling.
### How to resume
`Read MISSION.md, docs/PROGRESS.md, docs/DECISIONS.md and continue`. Jobs: `tmux ls`; completion = `results/runs/<id>/DONE`; failure = `exit.json` with non-zero code.
