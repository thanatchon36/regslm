"""Offline tests for the EUR-Lex adapter (fixtures) + one live smoke test."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from regslm.corpus.base import RawDoc, Unit
from regslm.corpus.eurlex import (
    INSTRUMENTS,
    EurLexAdapter,
    _CellarClient,
    _label_token,
)
from regslm.corpus.registry import ADAPTERS

FIX = Path(__file__).parent / "fixtures" / "eurlex"


@pytest.fixture
def adapter(tmp_path: Path) -> EurLexAdapter:
    return EurLexAdapter(raw_root=tmp_path, as_of=date(2026, 10, 15))


@pytest.fixture
def parsed(adapter: EurLexAdapter) -> tuple[object, list[Unit]]:
    raw = RawDoc(
        source="eurlex",
        url="http://publications.europa.eu/resource/cellar/x",
        sha256="0" * 64,
        path=str(FIX / "crr_trimmed.xhtml"),
        content_type="application/xhtml+xml",
        fetched_at=datetime.now(UTC),
    )
    return adapter.parse(raw)  # type: ignore[return-value]


def by_id(units: list[Unit]) -> dict[str, Unit]:
    return {u.unit_id: u for u in units}


def test_registered() -> None:
    assert ADAPTERS["eurlex"] is EurLexAdapter


def test_document_metadata(parsed: tuple[object, list[Unit]]) -> None:
    doc = parsed[0]
    assert doc.doc_id == "CRR"  # type: ignore[attr-defined]
    assert doc.version == "2026-06-26"  # type: ignore[attr-defined]
    assert doc.language == "en"  # type: ignore[attr-defined]
    assert "575/2013" in doc.title  # type: ignore[attr-defined]


def test_unit_ids_unique_and_prefixed(parsed: tuple[object, list[Unit]]) -> None:
    units = parsed[1]
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("eurlex/CRR/") for i in ids)
    assert all(u.version == "2026-06-26" and u.language == "en" for u in units)


def test_article_92_points_carry_leadin_in_heading(parsed: tuple[object, list[Unit]]) -> None:
    u = by_id(parsed[1])["eurlex/CRR/Art92/1/a"]
    assert u.text == "a Common Equity Tier 1 capital ratio of 4,5 %;"
    assert u.heading.startswith("Own funds requirements | Subject to Articles 93 and 94")
    assert u.heading.endswith("satisfy the following own funds requirements:")
    assert u.path[-3:] == ["Article 92", "1.", "(a)"]
    assert u.path[:2] == ["PART THREE", "TITLE I"]


def test_paragraph_without_points_is_unit(parsed: tuple[object, list[Unit]]) -> None:
    u = by_id(parsed[1])["eurlex/CRR/Art92/1a"]
    assert u.text.startswith("In addition to the requirement laid down in point (d)")
    assert u.heading == "Own funds requirements"


def test_definitions_nested_levels(parsed: tuple[object, list[Unit]]) -> None:
    ids = by_id(parsed[1])
    u = ids["eurlex/CRR/Art4/1/1/b/i"]
    assert u.text.startswith("the total value of the consolidated assets")
    assert "credit institution" in u.heading
    assert u.path[-4:] == ["1.", "(1)", "(b)", "(i)"]
    # lead-in is not duplicated into text
    assert "means an undertaking" not in u.text


def test_markup_footnotes_and_markers_removed(parsed: tuple[object, list[Unit]]) -> None:
    for u in parsed[1]:
        for s in (u.text, u.heading):
            assert not any(ch in s for ch in "►▼◄")
            assert "()" not in s
            assert "  " not in s
            assert "\xa0" not in s
    head = by_id(parsed[1])["eurlex/CRR/Art4/1/1/b/i"].heading
    assert "Directive 2014/65/EU of the European Parliament and of the Council," in head


def test_cross_references_kept_verbatim(parsed: tuple[object, list[Unit]]) -> None:
    u = by_id(parsed[1])["eurlex/CRR/Art92/4/g/i"]
    assert "Annex II" in u.text


def test_last_amended_from_header_table(parsed: tuple[object, list[Unit]]) -> None:
    la = {u.last_amended for u in parsed[1]}
    assert None not in la
    assert all(d is not None and d <= "2026-06-26" for d in la)
    # CRR III (Regulation (EU) 2024/1623 of 31 May 2024) last touched Art. 92 and Art. 4
    assert by_id(parsed[1])["eurlex/CRR/Art92/1/a"].last_amended == "2024-05-31"


def test_annex_items(parsed: tuple[object, list[Unit]]) -> None:
    u = by_id(parsed[1])["eurlex/CRR/AnnexII/1/a"]
    assert u.text == "single-currency interest rate swaps;"
    assert u.heading == "Types of derivatives | Interest-rate contracts:"
    assert u.path[0] == "ANNEX II"


def test_no_empty_or_reserved_units(parsed: tuple[object, list[Unit]]) -> None:
    assert all(u.text.strip() for u in parsed[1])
    assert not any(u.text.lower().strip("[]. ") in ("deleted", "reserved") for u in parsed[1])


def test_language_suffix_in_doc_id(adapter: EurLexAdapter, tmp_path: Path) -> None:
    src = (FIX / "crr_trimmed.xhtml").read_text(encoding="utf-8")
    fr = src.replace("32013R0575 — EN — 26.06.2026", "32013R0575 — FR — 26.06.2026")
    p = tmp_path / "fr.xhtml"
    p.write_text(fr, encoding="utf-8")
    raw = RawDoc(
        source="eurlex", url="u", sha256="1" * 64, path=str(p), content_type="x",
        fetched_at=datetime.now(UTC),
    )  # fmt: skip
    doc, units = adapter.parse(raw)
    assert doc.doc_id == "CRR-fr" and doc.language == "fr"
    assert units[0].unit_id.startswith("eurlex/CRR-fr/")


def test_label_tokens() -> None:
    assert _label_token("(a)") == "a"
    assert _label_token("1a.") == "1a"
    assert _label_token("-1.") == "m1"
    assert _label_token("— 2") == "d2"


def test_instruments() -> None:
    assert INSTRUMENTS["CRR"][0] == "32013R0575"
    assert set(INSTRUMENTS) == {"CRR", "CRD", "MiFID2", "AMLD4"}


def test_unknown_language_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EurLexAdapter(raw_root=tmp_path, languages=["xx"])


def _stub_sparql(adapter: EurLexAdapter, tmp_path: Path) -> None:
    body = tmp_path / "sparql.json"
    shutil.copy(FIX / "crr_sparql.json", body)
    raw = RawDoc(
        source="eurlex", url="s", sha256="2" * 64, path=str(body), content_type="application/json",
        fetched_at=datetime.now(UTC),
    )  # fmt: skip
    orig = adapter.client.get_negotiated

    def fake(url: str, headers: dict[str, str], suffix: str = "") -> RawDoc:
        return raw if "sparql" in url else orig(url, headers, suffix)

    adapter.client.get_negotiated = fake  # type: ignore[method-assign]


def test_consolidated_versions_skip_future(adapter: EurLexAdapter, tmp_path: Path) -> None:
    _stub_sparql(adapter, tmp_path)
    data = json.loads((FIX / "crr_sparql.json").read_text())
    dates = sorted((b["d"]["value"] for b in data["results"]["bindings"]), reverse=True)
    assert dates[0] > "2026-10-15"  # a not-yet-applicable consolidation exists in the fixture
    vs = adapter.consolidated_versions("32013R0575")
    assert vs[0][1] <= "2026-10-15"
    assert [d for _, d in vs] == sorted((d for _, d in vs), reverse=True)


def test_fetch_falls_back_when_latest_has_no_language(
    adapter: EurLexAdapter, tmp_path: Path
) -> None:
    _stub_sparql(adapter, tmp_path)
    vs = adapter.consolidated_versions("32013R0575")
    calls: list[str] = []
    orig = adapter.client.get_negotiated

    def fake(url: str, headers: dict[str, str], suffix: str = "") -> RawDoc:
        if "sparql" in url:
            return orig(url, headers, suffix)
        calls.append(url)
        if url == vs[0][0]:
            req = httpx.Request("GET", url)
            raise httpx.HTTPStatusError(
                "nf", request=req, response=httpx.Response(404, request=req)
            )
        return RawDoc(
            source="eurlex", url=url, sha256="3" * 64, path="p", content_type="x",
            fetched_at=datetime.now(UTC),
        )  # fmt: skip

    adapter.client.get_negotiated = fake  # type: ignore[method-assign]
    out = list(adapter.fetch(["CRR"]))
    assert len(out) == 1 and calls == [vs[0][0], vs[1][0]]


def test_cellar_client_blocked_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(202, text="")

    c = _CellarClient(cache_dir=tmp_path, source="eurlex", min_interval_s=0)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match=r"NEEDED\.md"):
        c.get_negotiated("http://example.org/x", {"Accept": "text/html"})


@pytest.mark.network
def test_live_amld4_smoke() -> None:
    a = EurLexAdapter()
    raw = next(a.fetch(["AMLD4"]))
    doc, units = a.parse(raw)
    assert doc.doc_id == "AMLD4" and len(units) > 300
    assert len({u.unit_id for u in units}) == len(units)
