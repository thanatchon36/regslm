import polars as pl

from regslm.corpus.sample import _allocate, has_trigger, stratified_gold


def _units() -> pl.DataFrame:
    rows = []
    for doc, n, lang in (("BIG", 900, "en"), ("SMALL", 100, "en"), ("TH", 200, "th")):
        for i in range(n):
            text = "Firms must report." if i % 3 else "'Firm' means a person."
            if lang == "th":
                text = "ให้ธนาคารรายงาน" if i % 3 else "นิยามคำว่า"
            rows.append(
                {
                    "unit_id": f"{doc}/{i}",
                    "doc_id": doc,
                    "language": lang,
                    "text": text,
                    "heading": "",
                }
            )
    return pl.DataFrame(rows)


def test_has_trigger() -> None:
    assert has_trigger("Institutions shall report")
    assert has_trigger("(b) appoint a chief risk officer;", "A bank must:")
    assert not has_trigger("'Credit institution' means an undertaking")
    assert has_trigger("ห้ามมิให้สถาบันการเงิน")


def test_allocate_sqrt_and_total() -> None:
    a = _allocate({"a": 900, "b": 100}, 40)
    assert sum(a.values()) == 40 and a["a"] == 30 and a["b"] == 10


def test_stratified_gold_shares_and_determinism() -> None:
    u = _units()
    g = stratified_gold(u, n=100, seed=1)
    assert g.height == 100
    assert (g["language"] == "th").sum() == 20
    assert 10 <= (~g["trigger"]).sum() <= 20
    assert g["unit_id"].n_unique() == 100
    assert g.equals(stratified_gold(u, n=100, seed=1))
