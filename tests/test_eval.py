import math

import polars as pl
import pytest

from regslm.eval import (
    MemRow,
    UnitResult,
    bootstrap_ci,
    cohen_kappa,
    crossover,
    field_f1,
    holm,
    inter_pass_agreement,
    memorisation_rate,
    obligation_prf,
    paired_bootstrap_diff,
    per_unit_counts,
    sanity_flags,
    seed_mean_ci,
    summarize,
)
from regslm.eval.metrics import metric
from regslm.eval.stats import t_quantile
from regslm.schema.obligation import (
    Citation,
    Extraction,
    Modality,
    Obligation,
    Span,
    Threshold,
)


def ob(start=0, end=10, mod=Modality.MUST, **kw):
    kw.setdefault("subject", "The firm")
    kw.setdefault("action", "report")
    return Obligation(
        modality=mod,
        citation=Citation(source="s", unit_id="u", span=Span(start=start, end=end)),
        **kw,
    )


def ex(*obs, uid="u"):
    return Extraction(unit_id=uid, obligations=list(obs))


def test_prf_known_counts():
    r = [
        UnitResult("a", ex(ob(0, 10), ob(20, 30)), ex(ob(0, 10), ob(50, 60))),  # tp1 fp1 fn1
        UnitResult("b", ex(ob(0, 10)), ex(ob(0, 10)), n_unresolved=1),  # tp1 fp1
        UnitResult("c", ex(ob(0, 10)), None),  # fn1 invalid
        UnitResult("d", ex(), ex()),  # nothing
    ]
    p = obligation_prf(r)
    assert (p.precision, p.recall) == (2 / 4, 2 / 4)
    assert p.f1 == 0.5
    df = per_unit_counts(r)
    s = summarize(df)
    assert s["validity_rate"] == 3 / 4
    assert s["hallucination_rate"] == 2 / 4
    assert s["empty_rate"] == 1 / 4
    assert df.filter(pl.col("unit_id") == "d")["tp"][0] == 0


def test_modality_mismatch_not_matched():
    p = obligation_prf([UnitResult("a", ex(ob(mod=Modality.MAY)), ex(ob()))])
    assert p.f1 == 0.0


def test_field_counts_and_numeric():
    gold = ob(
        condition="If X.",
        threshold=Threshold(value=5, unit="%"),
        deadline="within 30 days",
        exceptions=["A", "b"],
    )
    pred = ob(
        subject="the FIRM",
        condition="if y",
        threshold=Threshold(value=5, unit="percent"),
        deadline="in 30 days",
        exceptions=["B", "a"],
        object="extra",
    )
    unmatched_pred = ob(50, 60, condition="c")
    unmatched_gold = ob(70, 80, deadline="1 day")
    r = [UnitResult("a", ex(gold, unmatched_gold), ex(pred, unmatched_pred))]
    f = field_f1(r)
    assert (f["subject"].precision, f["subject"].recall) == (
        1 / 2,
        1 / 2,
    )  # tp1 (matched), fp1, fn1 unmatched
    df = per_unit_counts(r)
    row = df.row(0, named=True)
    assert (row["condition_tp"], row["condition_fp"], row["condition_fn"]) == (0, 2, 1)
    assert (row["threshold_tp"], row["threshold_fp"], row["threshold_fn"]) == (1, 0, 0)
    assert (row["deadline_tp"], row["deadline_fp"], row["deadline_fn"]) == (
        0,
        1,
        2,
    )  # strict: unequal
    assert (row["exceptions_tp"], row["object_fp"]) == (1, 1)
    assert row["num_n"] == 1 and row["num_ok"] == 1  # relaxed deadline equal
    assert row["exact"] == 0
    assert summarize(df)["numeric_exact"] == 1.0


def test_exact_and_numeric_failure():
    g = ob(threshold=Threshold(value=5))
    r = [
        UnitResult("a", ex(g), ex(g)),
        UnitResult("b", ex(g), ex(ob(threshold=Threshold(value=6)))),
    ]
    s = summarize(per_unit_counts(r))
    assert s["exact_obligation_rate"] == 0.5
    assert s["numeric_exact"] == 0.5


def test_empty_set_table():
    df = per_unit_counts([])
    assert df.height == 0
    assert summarize(df)["of1"] == 0.0


def _df(vals):
    return per_unit_counts(
        [UnitResult(f"u{i}", ex(ob()), ex(ob()) if v else ex()) for i, v in enumerate(vals)]
    )


def test_bootstrap_ci_deterministic_and_contains_point():
    df = _df([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    a = bootstrap_ci(df, metric("recall"), n=300, seed=3)
    b = bootstrap_ci(df, metric("recall"), n=300, seed=3)
    assert a == b
    pt, lo, hi = a
    assert pt == 0.7 and lo <= pt <= hi
    assert math.isnan(bootstrap_ci(df.clear(), metric("recall"), n=5)[1])


def test_paired_identical_is_zero():
    df = _df([1, 0, 1, 1, 0, 1])
    r = paired_bootstrap_diff(df, df.reverse(), metric("recall"), n=200)
    assert (r.diff, r.lo, r.hi, r.p) == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_diff(df, df.head(2), metric("recall"), n=5)
    e = paired_bootstrap_diff(df.clear(), df.clear(), metric("recall"), n=5)
    assert math.isnan(e.p)


def test_paired_detects_difference():
    a = _df([1] * 40 + [0] * 2)
    b = _df([0] * 40 + [1] * 2)
    r = paired_bootstrap_diff(a, b, metric("recall"), n=300)
    assert r.diff > 0.9 and r.lo > 0 and r.p < 0.01


def test_seed_mean_ci():
    m, _lo, hi = seed_mean_ci([0.5, 0.6, 0.7])
    assert m == pytest.approx(0.6)
    assert hi - m == pytest.approx(4.303 * 0.1 / math.sqrt(3))
    assert math.isnan(seed_mean_ci([1.0])[1])
    with pytest.raises(ValueError):
        seed_mean_ci([])


def test_t_quantile():
    assert t_quantile(0.975, 2) == 4.303
    assert t_quantile(0.975, 35) == 2.021
    assert t_quantile(0.975, 1000) == 1.960
    assert t_quantile(0.95, 10) == pytest.approx(1.812, abs=2e-3)
    for p, d in ((0.5, 3), (0.9, 0)):
        with pytest.raises(ValueError):
            t_quantile(p, d)


def test_holm_textbook():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.9]) == [0.9]
    assert holm([0.6, 0.7]) == [1.0, 1.0]


def test_crossover():
    ns = [500, 2000, 10000, 20000]
    ci = [(-1, -3, 1), (2, 0.5, 3), (3, 1, 5), (3, 1, 5)]
    assert crossover(ns, ci) == 2000
    assert crossover(ns, [(0, -1, 1), (0, -1, 1), (0, -1, 1), (3, 1, 5)]) is None  # last N alone
    assert crossover(ns, [(1, 1, 2), (0, -1, 1), (1, 1, 2), (1, 1, 2)]) == 10000
    assert crossover(list(reversed(ns)), list(reversed(ci))) == 2000


def test_sanity_flags():
    assert sanity_flags({"of1": 0.6, "field_f1.subject": 0.9}) == []
    fl = sanity_flags({"of1": 1.0, "recall": 0.0, "field_f1.action": 0.99})
    assert len(fl) == 3
    fl = sanity_flags(
        {"of1": 0.62, "numeric_exact": 0.5},
        parent={"of1": 0.60, "numeric_exact": 0.5},
        ci_width=0.01,
    )
    assert len(fl) == 1 and "beats parent" in fl[0]
    assert sanity_flags({"of1": 0.6}, parent={"of1": 0.6}, ci_width=0.01) == []


def test_kappa_textbook():
    a = [1] * 25 + [0] * 25  # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa([1, 1], [1, 1]) == 1.0
    assert cohen_kappa([1, 1], [1, 0]) == pytest.approx(0.0)
    assert cohen_kappa([1, 1], [0, 0]) == 0.0
    assert math.isnan(cohen_kappa([], []))
    with pytest.raises(ValueError):
        cohen_kappa([1], [])


def test_inter_pass_agreement():
    a = {
        "1": ex(ob(0, 10, condition="c"), ob(20, 30)),
        "2": ex(),
        "3": ex(ob(0, 10)),
        "x": None,
    }
    b = {
        "1": ex(ob(0, 10, condition="d"), ob(40, 50)),
        "2": ex(),
        "3": ex(ob(0, 10, mod=Modality.SHOULD)),
        "x": ex(),
    }
    g = inter_pass_agreement(a, b)
    assert g.n_units == 3 and g.n_matched == 1
    assert g.span_f1 == pytest.approx(1 / 3)  # tp1 fp2 fn2
    assert g.field_exact_agreement["condition"] == 0.0
    assert g.field_exact_agreement["subject"] == 1.0
    assert g.has_any_agreement == 1.0 and g.has_any_kappa == pytest.approx(1.0)
    empty = inter_pass_agreement({}, {})
    assert math.isnan(empty.has_any_agreement) and math.isnan(
        empty.field_exact_agreement["subject"]
    )


def test_memorisation():
    ref = ex(ob(0, 10, threshold=Threshold(value=7)), ob(20, 30, threshold=Threshold(value=9)))
    tg = [[0, "threshold.value", 5.0, 7.0], [1, "threshold.value", 3.0, 9.0], [0, "subject", 1, 2]]
    pred = ex(ob(0, 10, threshold=Threshold(value=5)), ob(20, 30, threshold=Threshold(value=9)))
    rows = [MemRow(ref, pred, tg), MemRow(ref, None, tg), MemRow(ref, ex(ob(0, 10)), tg)]
    m = memorisation_rate(rows)
    assert m["n_targets"] == 6
    assert m["memorisation_rate"] == 1 / 6
    assert m["perturbed_correct_rate"] == 1 / 6


def test_sanity_flags_any_metric_exact_zero_or_one() -> None:
    from regslm.eval.stats import sanity_flags

    flags = sanity_flags(
        {"of1": 0.71, "field_f1.threshold": 0.0, "validity_rate": 1.0, "field_f1.subject": 0.99}
    )
    assert any("field_f1.threshold" in f for f in flags)
    assert any("validity_rate" in f for f in flags)
    assert any("field_f1.subject" in f and "> 0.98" in f for f in flags)
    assert not any(f.startswith("of1") for f in flags)
