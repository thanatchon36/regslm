# Future work

Items deliberately left out of v0.1 to keep the budget on clean curves for the Qwen3 ladder
(MISSION §7), with the reason.

- **BIS Basel Framework and HKMA SPM as out-of-distribution evaluation sets** (tier L, local-only
  text). Adapters not built; evaluation inference would cost ~0.2 GPU-h per model. Add after the
  main grid if budget remains.
- **FCA Handbook (SYSC, COBS, SUP)** — OGL text, blocked on a sanctioned access route (owner).
  SYSC would become a second held-out rulebook.
- **PRA Rulebook, MAS Notices** — excluded by owner decision (D-031).
- **Gemma 3 1B/4B as a second family** — skipped by owner decision (D-039).
- **SEA-LION v3 8B** — optional block 8, run only if the measured budget allows.
- **Documentation site** (mkdocs) — README + docs/USAGE.md suffice for v0.1.
- **AML Act B.E. 2542 (Thai)** — not obtainable without violating robots.txt (docs/sources/NEEDED.md).
- **Apple M4 portability run** — `scripts/bench_mac.sh` exists; needs the owner's machine.
