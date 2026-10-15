"""Offline tests for the eCFR adapter (fixtures in tests/fixtures/ecfr) + one live smoke test."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from lxml import etree

from regslm.corpus.base import RawDoc, sha256_bytes
from regslm.corpus.ecfr import BULK, DEFAULT_SELECTION, VERSIONS, EcfrAdapter, _Nester, _Para

FIX = Path(__file__).parent / "fixtures" / "ecfr"
XML_URL = BULK.format(t=12)
VERS_URL = VERSIONS + "title-12.json?part=3"
XML_FIX = FIX / "ECFR-title12_part3_trimmed.xml"


def _raw(url: str, path: Path) -> RawDoc:
    return RawDoc(
        source="ecfr",
        url=url,
        sha256=sha256_bytes(path.read_bytes()),
        path=str(path),
        content_type="application/xml",
        fetched_at=datetime.now(UTC),
    )


@pytest.fixture()
def adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcfrAdapter:
    a = EcfrAdapter(raw_root=tmp_path)
    files = {
        XML_URL: XML_FIX,
        VERS_URL: FIX / "versions_title-12_part-3.json",
    }

    def fake_get(url: str, suffix: str = "") -> RawDoc:
        return _raw(url, files[url])  # KeyError => unexpected network request

    monkeypatch.setattr(a.client, "get", fake_get)
    return a


def test_urls_one_bulk_file_per_title(adapter: EcfrAdapter) -> None:
    urls = adapter.urls(["12CFR3", "17CFR240:240.15c3-1", "17CFR275", "12CFR217"])
    assert urls == [XML_URL, BULK.format(t=17)]
    assert len(adapter.urls()) == 2 and len(DEFAULT_SELECTION) == 12
    assert "/api/versioner/v1/full/" not in "".join(urls)  # disallowed by ecfr.gov robots.txt
    with pytest.raises(ValueError):
        adapter.urls(["nonsense"])


def test_parse_part3(adapter: EcfrAdapter) -> None:
    raw = _raw(XML_URL, XML_FIX)
    docs = adapter.parse_all(raw, ["12CFR3"])
    assert len(docs) == 1
    doc, units = docs[0]
    assert doc.doc_id == "12CFR3" and doc.version == "2026-10-01" and doc.language == "en"
    assert doc.sha256 == raw.sha256
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids)) > 20
    by_id = {u.unit_id: u for u in units}

    a = by_id["ecfr/12CFR3/§3.1/(a)"]
    assert a.text.startswith("Purpose. This part establishes minimum capital requirements")
    assert a.path == ["Part 3", "Subpart A", "§ 3.1", "(a)"]
    assert a.heading.startswith("§ 3.1 Purpose")
    assert a.last_amended == "2020-01-01"  # latest amendment_date <= version
    assert a.version == "2026-10-01"
    assert any(u.last_amended == "2026-04-01" for u in units if "§3.11" in u.unit_id)

    # inline sub-paragraph markers after a run-in heading: "(c)(3) <I>Risk-weighted assets.</I> (i) ..."
    nested = by_id["ecfr/12CFR3/§3.1/(c)/(3)/(i)"]
    assert "Risk-weighted assets" in nested.heading
    assert "(c) Applicability" in nested.heading
    # (f)(1)(iii)(B) -> italic (1)
    assert "ecfr/12CFR3/§3.1/(f)/(1)/(iii)/(B)/(1)" in by_id


def test_parse_single_doc(adapter: EcfrAdapter) -> None:
    adapter._selection = {12: {"3": None}}
    doc, units = adapter.parse(_raw(XML_URL, XML_FIX))
    assert doc.doc_id == "12CFR3" and units


def test_no_markup_reserved_or_footnote_noise(adapter: EcfrAdapter) -> None:
    _, units = adapter.parse_all(_raw(XML_URL, XML_FIX), ["12CFR3"])[0]
    for u in units:
        assert u.text == " ".join(u.text.split())
        assert "<" not in u.text and u.text
        assert "[Reserved]" not in u.text
    assert not any("3.4-3.9" in u.unit_id for u in units)  # reserved range => no unit


def test_definitions_anchor_nesting(adapter: EcfrAdapter) -> None:
    _, units = adapter.parse_all(_raw(XML_URL, XML_FIX), ["12CFR3"])[0]
    defs = [u for u in units if "§3.2/" in u.unit_id]
    assert any("/p" in u.unit_id and "/(1)" in u.unit_id for u in defs) or len(defs) > 5
    assert len({u.unit_id for u in defs}) == len(defs)


def _nest(markers: list[str]) -> list[str]:
    n = _Nester()
    out: list[str] = []
    for m in markers:
        italic = m.startswith("*")
        n.push(m.strip("*"), italic)
        out.append("".join(d for _, _, d in n.stack))
    return out


def test_nesting_letter_vs_roman_i() -> None:
    # (h) then (i): continues the alphabet when previous paragraph is not a lead-in
    assert _nest(["a", "b", "h", "i"])[-1] == "(i)"
    # (a)(1)(i)(ii) roman under a digit
    assert _nest(["a", "1", "i", "ii", "2", "b"]) == [
        "(a)",
        "(a)(1)",
        "(a)(1)(i)",
        "(a)(1)(ii)",
        "(a)(2)",
        "(b)",
    ]
    # (v) after (iv) is roman; (v) after (u) is a letter
    assert _nest(["a", "1", "i", "ii", "iii", "iv", "v"])[-1] == "(a)(1)(v)"
    assert _nest(["a", "1", "i", "A", "*1*", "*i*", "B"]) == [
        "(a)",
        "(a)(1)",
        "(a)(1)(i)",
        "(a)(1)(i)(A)",
        "(a)(1)(i)(A)(1)",
        "(a)(1)(i)(A)(1)(i)",
        "(a)(1)(i)(B)",
    ]


def test_nesting_ambiguous_i_uses_leadin() -> None:
    n = _Nester()
    for t in ("a", "g", "h"):
        n.push(t, False)
    n.prev_leadin = True
    n.push("i", False)
    assert [d for _, _, d in n.stack] == ["(h)", "(i)"]
    n2 = _Nester()
    for t in ("a", "g", "h"):
        n2.push(t, False)
    n2.push("i", False)
    assert [d for _, _, d in n2.stack] == ["(i)"]


def test_para_markers_and_italic_headings() -> None:
    p = _Para(
        etree.fromstring("<P>(a) <I>Heading.</I> (<I>1</I>) Text here.<FTREF/><SU>2</SU></P>")
    )
    assert p.markers == [("a", False, "Heading."), ("1", True, "")]
    assert p.text == "Text here."
    q = _Para(etree.fromstring("<P>(b) <I>Only heading.</I></P>"))
    assert q.heading_only
    r = _Para(etree.fromstring("<P>(c)(1)(i) Something.</P>"))
    assert [m[0] for m in r.markers] == ["c", "1", "i"]


def test_licence(adapter: EcfrAdapter) -> None:
    assert adapter.licence.redistributable and adapter.licence.derivative_ok
    assert json.loads((FIX / "versions_title-12_part-3.json").read_text())["content_versions"]


@pytest.mark.network
def test_live_smoke(tmp_path: Path) -> None:
    """Fetches the (cached) whole Title 17 bulk file and parses one small part."""
    a = EcfrAdapter(raw_root=tmp_path)
    raw = next(a.fetch(["17CFR275"]))
    ((doc, units),) = a.parse_all(raw, ["17CFR275"])
    assert doc.doc_id == "17CFR275" and units
    assert len({u.unit_id for u in units}) == len(units)
