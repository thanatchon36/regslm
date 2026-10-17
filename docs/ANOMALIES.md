# ANOMALIES

_None yet._

- **2026-10-17 — label_efficiency-qwen3-1.7b-lora-n20000-label-en_th-s1 crashed at step 995/1816** ("Unsloth: No or negligible GPU memory available"). Device memory (nvidia-smi, whole card) jumped from 9.2 GB to 11.6 GB at 10:44 and stayed there; the identical s0 config peaked at 8.9 GB, so the cause is most likely another process taking ~2.4 GB of VRAM, not the config. Classified as a crash, not "does not fit"; re-run once with the same seed (prereg §7).
  - **Correction (same day):** the re-run (11:36) crashed at the **same step 995** with the same error, so the cause is deterministic — a batch of long sequences in seed 1's data order needs more VRAM than the card has — not another process (the earlier jump was this run's own allocation). Per prereg §7 an OOM is a result: the cell is reported as "does not fit (OOM at step 995/1816)", the one re-run is used, and no config change is made. 1.7B N=20k has seeds 0 and 2.
