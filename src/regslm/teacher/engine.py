"""vLLM engine configuration and construction (lazy vLLM imports)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from regslm.schema.generation import GenExtraction

DEFAULT_MODEL = "Qwen/Qwen3-14B-AWQ"


class EngineConfig(BaseModel):
    model: str = DEFAULT_MODEL
    revision: str | None = None
    # 4096: prompt v5 is ~1.9k tokens; the prefix cache holds it once, so long units fit (D-042)
    max_model_len: int = 4096
    # 0.89 x 12,282 MiB = 10.9 GB, inside the 11 GB cap (D-014); needs the compositor < 1.3 GB.
    gpu_memory_utilization: float = 0.89
    max_num_batched_tokens: int = 1024
    max_num_seqs: int = 24  # prefix-cached system prompt → ~400 unique KV tokens per seq
    kv_cache_dtype: str = "fp8"
    attention_backend: str = "TRITON_ATTN"
    enforce_eager: bool = True  # CUDA graphs + torch.compile leave no KV room (D-015)
    # Few graph sizes: each captured size costs VRAM we do not have (D-015).
    cudagraph_capture_sizes: list[int] = [1, 2, 4, 8]
    enable_prefix_caching: bool = True
    # False → compact JSON grammar (no free whitespace). Qwen3-8B looped on whitespace until
    # max_tokens on 44 % of gold units with the default grammar (D-046).
    json_whitespace: bool = True
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0

    @property
    def model_slug(self) -> str:
        return self.model.replace("/", "--")

    def sampling_json(self) -> str:
        """Canonical JSON of everything that affects the sampled output."""
        d: dict[str, Any] = {k: getattr(self, k) for k in ("temperature", "max_tokens", "seed")}
        if not self.json_whitespace:  # only in the key when changed: existing caches stay valid
            d["json_whitespace"] = False
        return json.dumps(d, sort_keys=True)


CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def build_llm(cfg: EngineConfig) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "attention_backend": cfg.attention_backend,
        "enforce_eager": cfg.enforce_eager,
        "enable_prefix_caching": cfg.enable_prefix_caching,
        "max_num_seqs": cfg.max_num_seqs,
        "seed": cfg.seed,
    }
    if not cfg.json_whitespace:
        # Engine-level in vLLM 0.31: the per-request flag is ignored by the xgrammar backend.
        kwargs["structured_outputs_config"] = {
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        }
    if not cfg.enforce_eager:
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": cfg.cudagraph_capture_sizes}
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    return LLM(**kwargs)


def build_sampling(cfg: EngineConfig) -> Any:
    """SamplingParams with schema-constrained decoding from ``GenExtraction``."""
    from vllm import SamplingParams

    schema = GenExtraction.model_json_schema()
    try:
        from vllm.sampling_params import StructuredOutputsParams

        extra: dict[str, Any] = {
            "structured_outputs": StructuredOutputsParams(json=schema)
        }
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams  # type: ignore[attr-defined]

        extra = {"guided_decoding": GuidedDecodingParams(json=schema)}
    return SamplingParams(
        temperature=cfg.temperature, max_tokens=cfg.max_tokens, seed=cfg.seed, **extra
    )
