"""Inter-pass agreement between two gold annotation passes (docs/specs/eval.md).

Choices: units are those present in both passes (``None`` = no annotation, skipped).
Span F1 treats pass A as reference (it is symmetric for micro F1). Field kappa is on
*presence* (empty vs non-empty) over matched pairs; exact agreement per field is the share
of matched pairs with equal normalised values (both empty counts as agreement).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

from regslm.eval.metrics import FIELDS, _field_equal, _nonempty, prf
from regslm.rewards.normalize import match_obligations
from regslm.schema.obligation import Extraction, Obligation


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa for two equally long label sequences; nan if empty, 1.0 if both raters
    use a single identical category."""
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    if n == 0:
        return math.nan
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


@dataclass(frozen=True)
class Agreement:
    n_units: int
    span_f1: float
    n_matched: int
    field_presence_kappa: dict[str, float]
    field_exact_agreement: dict[str, float]
    modality_kappa: float
    has_any_agreement: float
    has_any_kappa: float


def inter_pass_agreement(
    pass_a: Mapping[str, Extraction | None], pass_b: Mapping[str, Extraction | None]
) -> Agreement:
    ids = sorted(
        u for u in pass_a.keys() & pass_b.keys() if pass_a[u] is not None and pass_b[u] is not None
    )
    tp = fp = fn = 0
    pairs_ab: list[tuple[Obligation, Obligation]] = []
    has_a: list[bool] = []
    has_b: list[bool] = []
    for u in ids:
        ea, eb = pass_a[u], pass_b[u]
        assert ea is not None and eb is not None
        oa, ob = ea.obligations, eb.obligations
        pairs = match_obligations(oa, ob)
        tp += len(pairs)
        fn += len(oa) - len(pairs)  # A is the reference: its unmatched obligations are FN
        fp += len(ob) - len(pairs)  # B's unmatched obligations are FP
        pairs_ab.extend((oa[i], ob[j]) for i, j in pairs)
        has_a.append(bool(oa))
        has_b.append(bool(ob))
    kappa = {
        f: cohen_kappa(
            [_nonempty(x, f) for x, _ in pairs_ab], [_nonempty(y, f) for _, y in pairs_ab]
        )
        for f in FIELDS
    }
    exact = {
        f: (sum(_field_equal(x, y, f) for x, y in pairs_ab) / len(pairs_ab))
        if pairs_ab
        else math.nan
        for f in FIELDS
    }
    return Agreement(
        n_units=len(ids),
        span_f1=prf(tp, fp, fn).f1,
        n_matched=len(pairs_ab),
        field_presence_kappa=kappa,
        field_exact_agreement=exact,
        modality_kappa=cohen_kappa(
            [x.modality.value for x, _ in pairs_ab], [y.modality.value for _, y in pairs_ab]
        ),
        has_any_agreement=(
            sum(x == y for x, y in zip(has_a, has_b, strict=True)) / len(ids) if ids else math.nan
        ),
        has_any_kappa=cohen_kappa(has_a, has_b),
    )
