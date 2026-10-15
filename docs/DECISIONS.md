# Design decisions

Format: ID — date — decision — why. Newest at the bottom.

- **D-001** — 2026-10-15 — Repo lives at `~/github/regslm` (Linux FS), not `~/regslm` as the owner's brief said. Same constraint (never under `/mnt/c`), existing checkout.
- **D-002** — 2026-10-15 — Two virtualenvs: `.venv` (core + vLLM serving/teacher) and `.venv-train` (Unsloth/TRL/PEFT). vLLM and Unsloth pin different torch builds; declared as a uv extras conflict. Make targets call `uv run --no-sync` so one env never silently re-syncs the other.
- **D-003** — 2026-10-15 — Obligation unit of output is `Extraction{unit_id, obligations[], rationale?}` per corpus unit; a unit with no obligation is an explicit empty list (needed for hallucination metrics and the `no_hallucinated_obligation` reward).
- **D-004** — 2026-10-15 — `citation.span` = half-open character offsets of the trigger sentence in the unit text (not token offsets): tokenizer-independent, so comparable across Qwen/Gemma/Typhoon and across languages.
- **D-005** — 2026-10-15 — Thresholds normalised to `{value: float, unit: str, comparator}`; one threshold per obligation (the binding one). Multiple thresholds → multiple obligations. Keeps numeric exact-match and perturbation checks well-defined.
- **D-006** — 2026-10-15 — Text fields are copied from the source (near-verbatim), not paraphrased; this makes span grounding checkable and keeps the student from learning teacher paraphrase style.
- **D-007** — 2026-10-15 — Teacher run with `kv_cache_dtype=fp8`, `max_model_len=4096`, eager mode: the Windows compositor holds ~1.3–1.5 GB VRAM, leaving ~10.8 GB; 14B-AWQ weights take ~9.4 GB, so KV room is the bottleneck (see results/runs/teacher-smoke).
- **D-008** — 2026-10-15 — Qwen3 thinking mode disabled for the teacher (`enable_thinking=False`); the "label + rationale" variant asks for an explicit short `rationale` field instead, so rationale length is bounded and token budgets are comparable (RQ2).
