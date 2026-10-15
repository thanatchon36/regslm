# Using `regslm extract`

> **Machine extraction — review by a qualified person before any use.** Output is a research
> artefact, not legal advice. Every run prints this notice to stderr and records it in the
> final `{"_summary": ...}` line.

```bash
regslm fetch pra fca ecfr            # optional: public regulatory text
regslm extract --model thanatchon36/regslm-qwen3-1.7b-oblig-awq doc.pdf -o doc.jsonl
```

Models live under `thanatchon36/regslm-qwen3-<size>-oblig[-awq|-gguf]` (D-032).
Accepted inputs: `.pdf`, `.html`/`.htm`, `.txt`, `.md`. Options: `--backend vllm|llamacpp`,
`--format bf16|awq|gguf` (backend and format are inferred from each other and from the model
name), `--language auto|en|th` (auto = Thai if more than 30 % of characters are Thai),
`--guided` (vLLM schema-constrained decoding), `-o out.jsonl` (default stdout).

## Examples

English rule PDF, GPU:

```bash
regslm extract --model thanatchon36/regslm-qwen3-1.7b-oblig-awq --format awq \
  rule.pdf -o rule.jsonl
# 212 units, 37 obligations, 1 invalid, 0 unresolved quotes. machine extraction — ...
```

Thai notification PDF on CPU with llama.cpp (needs `llama-server` on `PATH`, or
`LLAMA_SERVER_DIR`; use `--gpu` to offload layers):

```bash
regslm extract --backend llamacpp --format gguf \
  --model thanatchon36/regslm-qwen3-1.7b-oblig-gguf --language th notice.pdf -o notice.jsonl
```

## Output

One JSON line per obligation (schema: `regslm schema`), plus `quote` (the text at the span)
and `unit_text`. `citation.span` is a character range into `unit_text`; `citation.unit_id`
is `doc/<n>` or `doc/<heading-path>/<n>`, e.g. `doc/Article5/2` or `doc/มาตรา5/1`. List items
inherit the preceding paragraph ending in ":" as their lead-in. Paragraphs over 2000
characters are split at sentence ends.

The last line is `{"_summary": {"units", "obligations", "invalid", "unresolved",
"invalid_unit_ids", "disclaimer"}}`: `invalid` counts units whose output was not valid
JSON for the schema (no obligations are emitted for them); `unresolved` counts obligations
dropped because their quote could not be located in the unit.

## Hardware (TODO: fill from `results/hardware.json` and `results/bench/`)

| Model | Format | Fits | Throughput | Notes |
|---|---|---|---|---|
| 0.6B | bf16 / awq / gguf | TODO | TODO | TODO |
| 1.7B | awq | TODO GB VRAM | TODO units/s | TODO |
| 1.7B | gguf Q4_K_M | CPU, TODO GB RAM | TODO units/s | TODO |
| 4B | awq | TODO | TODO | TODO |
