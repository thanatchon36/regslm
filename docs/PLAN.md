# Plan

## Phase 1 — Corpus, schema, gold/silver, literature
1. Hardware checks → `results/hardware.json` ✅
2. Skeleton: uv project, Makefile (tmux jobs), docs, schema, corpus contracts ✅
3. Licence terms per source → `docs/sources/LICENSES.md` (subagent research → main-thread decision)
4. Adapters (one Sonnet subagent each, spec in `docs/specs/`): eCFR, EUR-Lex → BIS → PRA/FCA (local-only text) → HKMA → MAS (likely NEEDED.md) → Thai (Royal Gazette / Krisdika)
5. `regslm fetch | parse | snapshot` → `documents/units/manifest.parquet`, `data/corpus/VERSION`
6. Annotation guide (main thread) with ≥30 worked examples
7. Teacher feasibility (VRAM) → teacher module (vLLM, content-hash cache, two prompt variants)
8. Gold: 300 stratified units, two passes (14B teacher + 8B, different prompts) → `data/gold/to_adjudicate.jsonl`
9. Silver: 30–50k units, ≤12 GPU-h, label-only + rationale
10. Splits incl. held-out rulebook, perturbed test, post-cutoff slice
11. Literature review (one subagent per cluster) → `docs/literature/`

## Phase 2 — Pre-registration (`prereg-v1` tag) — later
## Phase 3–6 — see MISSION.md

## Budget
`uv run --no-sync python experiments/budget.py` projects GPU-hours from config/grid.yaml; re-run before each priority block (D-035).
