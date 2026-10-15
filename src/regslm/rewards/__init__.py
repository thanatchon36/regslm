"""Verifiable GRPO rewards for obligation extraction (paper §4.4; docs/specs/rewards.md)."""

from regslm.rewards.components import (
    COMPONENTS,
    citation_resolves,
    deadline_exact,
    no_hallucinated_obligation,
    numeric_exact,
    parse,
    schema_valid,
    span_grounded,
)
from regslm.rewards.composite import (
    DEFAULT_WEIGHTS,
    RewardBreakdown,
    batch_stats,
    load_weights,
    reward,
)
from regslm.rewards.normalize import match_obligations, normalize_text

__all__ = [
    "COMPONENTS",
    "DEFAULT_WEIGHTS",
    "RewardBreakdown",
    "batch_stats",
    "citation_resolves",
    "deadline_exact",
    "load_weights",
    "match_obligations",
    "no_hallucinated_obligation",
    "normalize_text",
    "numeric_exact",
    "parse",
    "reward",
    "schema_valid",
    "span_grounded",
]
