"""Evaluation metrics and statistics (paper §4.5; definitions in docs/RESEARCH_QUESTIONS.md §0).

- :mod:`regslm.eval.metrics`: obligation P/R/OF1, field F1, numeric-exact, memorisation rate,
  per-unit count tables.
- :mod:`regslm.eval.stats`: unit bootstrap, paired bootstrap, seed t-intervals, Holm, RQ1
  crossover rule, sanity flags.
- :mod:`regslm.eval.agreement`: inter-pass agreement of gold annotation passes.

See docs/specs/eval.md.
"""

from regslm.eval.agreement import Agreement, cohen_kappa, inter_pass_agreement
from regslm.eval.metrics import (
    FIELDS,
    PRF,
    MemRow,
    UnitResult,
    field_f1,
    memorisation_rate,
    obligation_prf,
    per_unit_counts,
    summarize,
)
from regslm.eval.stats import (
    PairedResult,
    bootstrap_ci,
    crossover,
    holm,
    paired_bootstrap_diff,
    sanity_flags,
    seed_mean_ci,
)

__all__ = [
    "FIELDS",
    "PRF",
    "Agreement",
    "MemRow",
    "PairedResult",
    "UnitResult",
    "bootstrap_ci",
    "cohen_kappa",
    "crossover",
    "field_f1",
    "holm",
    "inter_pass_agreement",
    "memorisation_rate",
    "obligation_prf",
    "paired_bootstrap_diff",
    "per_unit_counts",
    "sanity_flags",
    "seed_mean_ci",
    "summarize",
]
