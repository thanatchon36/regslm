# Progress

## Waiting on owner
- **WSL memory/swap below target.** `free -g` shows 15 GB RAM / 4 GB swap; mission requires `memory=24GB` and `swap=16GB`. Add to `C:\Users\<you>\.wslconfig` under `[wsl2]` (keep `networkingMode=mirrored`), then `wsl --shutdown` from PowerShell. Until then: teacher/training run but full-FT 1.7B and GGUF conversion of 8B may hit host-RAM limits.
- **Windows sleep.** Set Windows power plan to never sleep while long jobs run (cannot be controlled from WSL).
- **Hugging Face username** for releases — brief contained a placeholder. Only needed in Phase 5.
- **PRA Rulebook and MAS Notices — licence decision.** Terms do not grant redistribution or derivative use (evidence: docs/sources/LICENSES_research.md). Current policy: excluded entirely (tier X). Options: (a) keep excluded [default]; (b) allow local-only evaluation use like BIS/HKMA. Reply with a or b.
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
- `teacher-bench-label` (tmux): 200 EUR-Lex units, label variant → throughput for the 12 GPU-h budget plan. Check `results/runs/teacher-bench-label/metrics.json`.
- Subagents: Thai adapter (HF `pythainlp/thailaw` + BoT), corpus snapshot CLI.
### Next
1. Teacher throughput → decide silver N (D-015 plan); label silver (label variant) in tmux.
2. Snapshot → stratified gold sample (300) → gold pass A (14B) + pass B (Qwen3-8B, different prompt) → `data/gold/to_adjudicate.jsonl`.
3. BIS / HKMA adapters (tier L, eval only); splits incl. held-out + perturbed + post-cutoff.
4. Finalise and tag `prereg-v1`; then `make quick`.
### How to resume
`Read MISSION.md, docs/PROGRESS.md, docs/DECISIONS.md and continue`. Jobs: `tmux ls`; completion = `results/runs/<id>/DONE`; failure = `exit.json` with non-zero code.
