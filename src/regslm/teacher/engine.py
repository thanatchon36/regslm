"""vLLM engine configuration and construction (lazy vLLM imports)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from regslm.schema import Extraction

DEFAULT_MODEL = "Qwen/Qwen3-14B-AWQ"


class EngineConfig(BaseModel):
    model: str = DEFAULT_MODEL
    revision: str | None = None
    max_model_len: int = 3072
    gpu_memory_utilization: float = 0.87
    max_num_batched_tokens: int = 1024
    max_num_seqs: int = 8
    kv_cache_dtype: str = "fp8"
    attention_backend: str = "TRITON_ATTN"
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0

    @property
    def model_slug(self) -> str:
        return self.model.replace("/", "--")

    def sampling_json(self) -> str:
        """Canonical JSON of everything that affects the sampled output."""
        d = {k: getattr(self, k) for k in ("temperature", "max_tokens", "seed")}
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
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    return LLM(**kwargs)


def build_sampling(cfg: EngineConfig) -> Any:
    """SamplingParams with schema-constrained decoding from ``Extraction``."""
    from vllm import SamplingParams

    schema = Extraction.model_json_schema()
    try:
        from vllm.sampling_params import StructuredOutputsParams

        extra = {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        extra = {"guided_decoding": GuidedDecodingParams(json=schema)}
    return SamplingParams(
        temperature=cfg.temperature, max_tokens=cfg.max_tokens, seed=cfg.seed, **extra
    )
