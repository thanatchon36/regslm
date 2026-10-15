"""Feasibility probe: does Qwen3-14B-AWQ fit in the 11 GB budget with useful KV room?

Not part of the library; records results/runs/teacher-smoke/metrics.json.
"""

import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

try:
    from vllm.sampling_params import StructuredOutputsParams as SO

    def so(schema: dict) -> dict:
        return {"structured_outputs": SO(json=schema)}
except ImportError:  # older vLLM
    from vllm.sampling_params import GuidedDecodingParams as SO

    def so(schema: dict) -> dict:
        return {"guided_decoding": SO(json=schema)}

from regslm.schema import Extraction

OUT = Path("results/runs/teacher-smoke")
UNITS = [
    ("demo/1", "Institutions shall at all times satisfy a Common Equity Tier 1 capital ratio of 4.5 %."),
    ("demo/2", "A covered company must file the report with the Board within 30 days after the end of each calendar quarter, unless the Board grants an extension."),
    ("demo/3", "For the purposes of this Part, 'eligible capital' means the sum of Tier 1 capital and Tier 2 capital."),
    ("demo/4", "A firm may, with the prior permission of the competent authority, use internal models to calculate its own funds requirements."),
] * 2


def main() -> None:
    t0 = time.time()
    llm = LLM(
        model="Qwen/Qwen3-14B-AWQ",
        max_model_len=4096,
        gpu_memory_utilization=0.86,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        max_num_seqs=16,
    )
    load_s = time.time() - t0
    schema = Extraction.model_json_schema()
    sp = SamplingParams(temperature=0.0, max_tokens=768, **so(schema))
    msgs = [
        [
            {"role": "system", "content": "Extract every legal obligation from the unit as JSON matching the schema. Definitions are not obligations. citation.span is the character offset [start,end) of the trigger sentence in the unit text."},
            {"role": "user", "content": f"source=demo unit_id={uid}\n\n{text}"},
        ]
        for uid, text in UNITS
    ]
    t1 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    gen_s = time.time() - t1
    valid = 0
    samples = []
    for o in outs:
        txt = o.outputs[0].text
        try:
            Extraction.model_validate_json(txt)
            valid += 1
        except Exception as e:
            txt += f"  <<INVALID: {e!r}"[:300]
        samples.append(txt[:600])
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    cfg = llm.llm_engine.vllm_config.cache_config
    metrics = {
        "load_s": round(load_s, 1), "gen_s": round(gen_s, 2), "n": len(outs), "valid": valid,
        "gen_tokens": ntok, "tok_per_s": round(ntok / gen_s, 1),
        "num_gpu_blocks": getattr(cfg, "num_gpu_blocks", None), "block_size": cfg.block_size,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
