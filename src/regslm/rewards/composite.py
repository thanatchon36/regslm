"""Composite GRPO reward with per-component breakdown (docs/specs/rewards.md)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from regslm.corpus.base import Unit
from regslm.rewards.components import COMPONENTS, resolved
from regslm.rewards.normalize import match_obligations
from regslm.schema.obligation import Extraction

#: Precision-type components averaged inside the gate, in spec order.
PRECISION = ("cite", "ground", "num", "dl", "hall")
REFERENCE_FREE = ("cite", "ground")
DEFAULT_WEIGHTS: dict[str, float] = {
    "schema": 1.0, "cite": 1.0, "ground": 1.0, "num": 1.0, "dl": 0.5, "hall": 1.0, "rec": 1.5,
}  # fmt: skip


@dataclass(frozen=True)
class RewardBreakdown:
    """Total reward plus each component (``schema``, ``cite``, ``ground``, ``num``, ``dl``,
    ``hall``), the separately reported ``recall`` (unweighted fraction) and ``recall_term``."""

    total: float
    components: dict[str, float] = field(default_factory=dict)
    recall: float = 0.0
    recall_term: float = 0.0
    empty: bool = False


def load_weights(source: Mapping[str, float] | str | Path | None = None) -> dict[str, float]:
    """Weights from a dict or YAML path (missing keys take defaults; unknown keys raise)."""
    raw: Mapping[str, Any]
    if source is None:
        raw = {}
    elif isinstance(source, Mapping):
        raw = source
    else:
        loaded = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        raw = loaded or {}
    unknown = set(raw) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown reward weights: {sorted(unknown)}")
    return {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in raw.items()}}


def reward(
    completion: str,
    unit: Unit,
    reference: Extraction | None,
    weights: Mapping[str, float] | str | Path | None = None,
) -> RewardBreakdown:
    """R = w_schema*schema * sum(w_i c_i)/sum(w_i) + w_rec*matched/max(1, n_ref).

    Without a reference only the reference-free components (cite, ground) are averaged
    and there is no recall term.
    """
    w = load_weights(weights)
    comps = {k: fn(completion, unit, reference) for k, fn in COMPONENTS.items()}
    names = PRECISION if reference is not None else REFERENCE_FREE
    wsum = sum(w[k] for k in names)
    inner = sum(w[k] * comps[k] for k in names) / wsum if wsum else 0.0
    total = w["schema"] * comps["schema"] * inner
    r = resolved(completion, unit)
    recall = 0.0
    if r is not None and reference is not None:
        matched = len(match_obligations(r.extraction.obligations, reference.obligations))
        recall = matched / max(1, len(reference.obligations))
    recall_term = w["rec"] * recall
    return RewardBreakdown(
        total=total + recall_term,
        components=comps,
        recall=recall,
        recall_term=recall_term,
        empty=r is not None and not r.extraction.obligations and not r.unresolved,
    )


def batch_stats(
    completions: Sequence[str], breakdowns: Sequence[RewardBreakdown]
) -> dict[str, float]:
    """Per-step logging: each component's batch mean, recall, empty fraction, mean length."""
    n = max(1, len(breakdowns))
    stats = {f"reward/{k}": sum(b.components[k] for b in breakdowns) / n for k in COMPONENTS}
    stats["reward/recall"] = sum(b.recall for b in breakdowns) / n
    stats["reward/total"] = sum(b.total for b in breakdowns) / n
    stats["reward/empty_frac"] = sum(b.empty for b in breakdowns) / n
    stats["reward/mean_len"] = sum(len(c) for c in completions) / max(1, len(completions))
    return stats
