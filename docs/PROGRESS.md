# Progress

## Waiting on owner
- **WSL memory/swap below target.** `free -g` shows 15 GB RAM / 4 GB swap; mission requires `memory=24GB` and `swap=16GB`. Add to `C:\Users\<you>\.wslconfig` under `[wsl2]` (keep `networkingMode=mirrored`), then `wsl --shutdown` from PowerShell. Until then: teacher/training run but full-FT 1.7B and GGUF conversion of 8B may hit host-RAM limits.
- **Windows sleep.** Set Windows power plan to never sleep while long jobs run (cannot be controlled from WSL).
- **Hugging Face username** for releases — brief contained a placeholder. Only needed in Phase 5.
- **pandoc** not installed (`sudo apt install pandoc texlive-xetex`) — only needed for the paper PDF (Phase 5).

## Phase 1 — in progress
### Shipped
- Hardware report (`results/hardware.json`): RTX 4070 Ti 12 GB (CC 8.9, ~1.3–1.5 GB held by Windows), i5-13500, vLLM 0.31.0 + torch 2.13/cu130, FP8 supported.
- Skeleton, schema (`src/regslm/schema`), corpus contracts (`src/regslm/corpus/base.py`), tmux job runner.
### Running
- Check with `tmux ls`; logs in `results/runs/<id>/train.log`; completion = `results/runs/<id>/DONE`.
### Open
- See docs/PLAN.md Phase 1 list.
