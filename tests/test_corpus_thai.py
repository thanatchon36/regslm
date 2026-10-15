"""Offline tests for the Thai adapter (fixtures) + one live smoke test."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from regslm.corpus import thai
from regslm.corpus.base import Document, RawDoc, Unit
from regslm.corpus.registry import ADAPTERS

FIX = Path(__file__).parent / "fixtures" / "thai"


@pytest.fixture
def adapter(tmp_path: Path) -> thai.ThaiAdapter:
    return thai.ThaiAdapter(raw_root=tmp_path)


def _raw(url: str, path: Path) -> RawDoc:
    return RawDoc(
        source="thai",
        url=url,
        sha256="0" * 64,
        path=str(path),
        content_type="x",
        fetched_at=datetime.now(UTC),
    )


def parse_fixture(
    adapter: thai.ThaiAdapter, doc_id: str, fixture: str
) -> tuple[Document, list[Unit]]:
    spec = thai.SPECS[doc_id]
    url = spec.url + "#__index_level_0__=1" if spec.route == "hf" else spec.url
    return adapter.parse(_raw(url, FIX / fixture))


@pytest.fixture
def fiba_th(adapter: thai.ThaiAdapter) -> tuple[Document, list[Unit]]:
    return parse_fixture(adapter, "FIBA2551", "fiba_th_trimmed.pdf")


@pytest.fixture
def fiba_en(adapter: thai.ThaiAdapter) -> tuple[Document, list[Unit]]:
    return parse_fixture(adapter, "FIBA2551-en", "fiba_en_trimmed.pdf")


def by_id(units: list[Unit]) -> dict[str, Unit]:
    return {u.unit_id: u for u in units}


def test_registered() -> None:
    assert ADAPTERS["thai"] is thai.ThaiAdapter
    assert thai.ThaiAdapter.licence.redistributable


def test_catalog() -> None:
    assert {"SEA2535", "PSA2560", "FIBA2551", "BOTA2485", "FIBA2551-en"} <= set(thai.SPECS)
    notices = [s for s in thai.SPECS.values() if s.kind == "notice"]
    assert 18 <= len(notices) <= 30
    assert all(s.url.startswith("https://www.bot.or.th/content/dam/bot/") for s in notices)
    assert thai.SPECS["FIBA2551-en"].parallel_of == "FIBA2551"


def test_hf_statute_sea(adapter: thai.ThaiAdapter) -> None:
    doc, units = parse_fixture(adapter, "SEA2535", "sea2535_extract_trimmed.json")
    assert doc.doc_id == "SEA2535" and doc.language == "th"
    assert "blob/4c1e2c091b15b5f01d836d5a23398594157e021c" in doc.url
    assert doc.version == thai.HF_REVISION_DATE
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("thai/SEA2535/") for i in ids)
    u = by_id(units)
    s1 = u["thai/SEA2535/s1/p1"]
    assert s1.text.startswith("พระราชบัญญัตินี้เรียกว่า")
    assert "ํ" not in s1.text  # nikhahit + sara aa joined to sara am
    assert s1.path == ["s1", "p1"]
    # section number "๔/๑" -> ASCII token s4_1, Thai digits kept in text elsewhere
    assert "thai/SEA2535/s4_1/p1" in u
    # items carry the lead-in in heading, not in text
    item = u["thai/SEA2535/s3/p1/i1"]
    assert item.heading == "ให้ยกเลิก"
    assert item.text.startswith("พระราชบัญญัติตลาดหลักทรัพย์")
    assert item.last_amended is None
    # preamble is skipped
    assert not any("พระบรมราชโองการ" in x.text for x in units)


def test_hf_statute_psa_soft_wraps_and_quotes(adapter: thai.ThaiAdapter) -> None:
    _, units = parse_fixture(adapter, "PSA2560", "psa2560_extract_trimmed.json")
    assert all("\r" not in x.text and "\n" not in x.text for x in units)
    assert not any("[" in x.text for x in units)  # footnote markers removed
    s1 = by_id(units)["thai/PSA2560/s1/p1"]
    assert s1.text == "พระราชบัญญัตินี้เรียกว่า “พระราชบัญญัติระบบการชำระเงิน พ.ศ. ๒๕๖๐”"
    # each definition is a paragraph (วรรค) of section 3 with its own unit
    assert by_id(units)["thai/PSA2560/s3/p2"].text.startswith("“ระบบการชำระเงิน” หมายความว่า")


def test_pdf_statute_th_repairs_and_items(fiba_th: tuple[Document, list[Unit]]) -> None:
    doc, units = fiba_th
    assert doc.doc_id == "FIBA2551" and doc.language == "th"
    assert doc.version == "2015-01-01"  # latest amendment year found in the fixture footnotes
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids))
    u = by_id(units)
    assert u["thai/FIBA2551/s2/p1"].text.startswith("พระราชบัญญัตินี้ให้ใช้บังคับ")
    assert "[" not in u["thai/FIBA2551/s2/p1"].text  # footnote marker removed
    text = " ".join(x.text for x in units)
    assert "กำร" not in text and "การ" in text  # Council-of-State า/ำ swap reversed
    leaf = u["thai/FIBA2551/s48/p1/i1/k3"]  # (ค) is the third item (letters skip ฃ ฅ)
    assert leaf.path == ["pt4", "s48", "p1", "i1", "k3"]
    assert leaf.heading.startswith("ส่วนที่ ๔ ข้อห้ามในการให้สินเชื่อ | ภายใต้บังคับมาตรา ๕๙")
    assert leaf.heading.endswith("หรือผู้ที่เกี่ยวข้องกับบุคคลดังกล่าว เว้นแต่")
    assert leaf.last_amended == "2015-01-01"  # footnote: amended by the 2558 (2015) Act
    assert u["thai/FIBA2551/s3/p1/i2"].last_amended is None
    # watermark never leaks
    assert "กฤษฎีกา" not in text


def test_pdf_statute_en_aligns_with_th(
    fiba_th: tuple[Document, list[Unit]], fiba_en: tuple[Document, list[Unit]]
) -> None:
    doc, en = fiba_en
    assert doc.doc_id == "FIBA2551-en" and doc.language == "en"
    assert all(u.language == "en" and u.unit_id.startswith("thai/FIBA2551-en/") for u in en)
    assert by_id(en)["thai/FIBA2551-en/s3/p1/i1"].heading == "The following shall be repealed:"
    pairs = thai.align_parallel(fiba_th[1], en)
    assert ("thai/FIBA2551/s1/p1", "thai/FIBA2551-en/s1/p1") in pairs
    assert ("thai/FIBA2551/s3/p1/i2", "thai/FIBA2551-en/s3/p1/i2") in pairs
    assert len({a for a, _ in pairs}) == len(pairs)


def test_notice_clauses(adapter: thai.ThaiAdapter) -> None:
    doc, units = parse_fixture(adapter, "BOTN25550332", "botn25550332_trimmed.pdf")
    assert doc.doc_id == "BOTN25550332" and doc.version == "2012-01-01"
    ids = [u.unit_id for u in units]
    assert len(ids) == len(set(ids)) and ids[0].startswith("thai/BOTN25550332/c5")
    u = by_id(units)
    d = u["thai/BOTN25550332/c5.1/p2"]
    assert d.text.startswith("“ธนาคารพาณิชย์” หมายความว่า")
    assert d.heading == "5 เนื้อหา > 5.1 คำจำกัดความ"  # short numbered clauses act as headings
    assert d.path == ["c5.1", "p2"]


def test_segmenter_chapters_parts_items() -> None:
    text = "\n".join(
        [
            "พระราชบัญญัติ",
            "มาตรา ๑ พระราชบัญญัตินี้เรียกว่า “ก”",
            "หมวด ๑ บททั่วไป",
            "ส่วนที่ ๒ การกำกับ",
            "มาตรา ๒ ให้ผู้ประกอบธุรกิจปฏิบัติดังต่อไปนี้",
            "(๑) จัดทำรายงาน",
            "(ก) รายงานประจำปี",
            "(ข) รายงานรายไตรมาส",
            "(๒) ส่งข้อมูล",
            "ในกรณีตามวรรคหนึ่ง ให้ปฏิบัติโดยเร็ว",
            "มาตรา ๒ ทวิ ให้เป็นไปตามประกาศ",
            "มาตรา ๒/๑ ยกเลิก",
            "มาตรา ๓ ข้อความที่ขึ้นต้นด้วยมาตรา ๑ ในวรรคนี้",
            "ผู้รับสนองพระบรมราชโองการ",
            "มาตรา ๔ ไม่ถูกอ่าน",
        ]
    )
    spec = thai.SPECS["SEA2535"]
    seg = thai._ActSegmenter(spec, {}, "2024-02-25", thai._STOP_ACT_RE)
    seg.run(thai.blocks_from_text(text).blocks)
    ids = [u.unit_id for u in seg.units]
    assert ids == [
        "thai/SEA2535/s1/p1",
        "thai/SEA2535/s2/p1/i1/k1",
        "thai/SEA2535/s2/p1/i1/k2",
        "thai/SEA2535/s2/p1/i2",
        "thai/SEA2535/s2/p2",
        "thai/SEA2535/s2bis/p1",
        "thai/SEA2535/s3/p1",
    ]
    k1 = seg.units[1]
    assert k1.path == ["ch1", "pt2", "s2", "p1", "i1", "k1"]
    assert k1.heading == "หมวด ๑ บททั่วไป > ส่วนที่ ๒ การกำกับ | ให้ผู้ประกอบธุรกิจปฏิบัติดังต่อไปนี้ | จัดทำรายงาน"
    assert seg.units[-1].text.startswith("ข้อความที่ขึ้นต้นด้วยมาตรา ๑")  # not a new section


def test_pua_and_garble_detection() -> None:
    assert thai._fix_pua("แหง ให") == "แห่ง ให้"
    good = "ธนาคารพาณิชย์ต้องดำรงเงินกองทุนตามที่ธนาคารแห่งประเทศไทยกำหนด " * 8
    assert not thai.page_is_garbled(good)
    no_tones = good.replace("่", "").replace("้", "") + "ให้ได้ " * 10
    no_tones = no_tones.replace("ให้", "ให").replace("ได้", "ได")
    assert thai.page_is_garbled(no_tones)
    junk = "/afiiotraOf/fpSbf/afiiotpOof Ba788sec1s;Ba788sec30 " * 12 + "ก" * 10
    assert thai.page_is_garbled(junk)


def test_amend_year() -> None:
    assert thai._amend_year("มาตรา ๔๘ แก้ไขเพิ่มเติมโดยพระราชบัญญัติ (ฉบับที่ ๒) พ.ศ. ๒๕๕๘") == 2015
    assert thai._amend_year("ราชกิจจานุเบกษา เล่ม ๑๒๕ ตอนที่ ๒๗ ก ๒๕๕๑") is None
    assert thai._amend_year("As added by Section 3 of the FIBA (No.3), B.E. 2561 (2018)") == 2018


def test_unavailable_and_unknown(adapter: thai.ThaiAdapter) -> None:
    with pytest.raises(RuntimeError, match=r"NEEDED\.md"):
        list(adapter.fetch(["AMLA2542"]))
    with pytest.raises(KeyError):
        list(adapter.fetch(["NOPE"]))


def test_fetch_uses_cache_offline(adapter: thai.ThaiAdapter) -> None:
    spec = thai.SPECS["FIBA2551-en"]
    key = adapter.client._key(spec.url)
    body = adapter.client.cache_dir / f"{key}.pdf"
    shutil.copy(FIX / "fiba_en_trimmed.pdf", body)
    raw = RawDoc(
        source="thai",
        url=spec.url,
        sha256="1" * 64,
        path=str(body),
        content_type="application/pdf",
        fetched_at=datetime.now(UTC),
    )
    (adapter.client.cache_dir / f"{key}.meta.json").write_text(raw.model_dump_json())
    got = list(adapter.fetch(["FIBA2551-en"]))  # served from the on-disk cache
    assert got[0].url == spec.url and got[0].sha256 == "1" * 64
    doc, units = adapter.parse(got[0])
    assert doc.doc_id == "FIBA2551-en" and units


def test_hf_row_selection_picks_longest_matching_row(adapter: thai.ThaiAdapter) -> None:
    spec = thai.SPECS["PSA2560"]
    shard = thai.HF_SHARDS[1]
    url = f"{thai.HF_BASE}/resolve/{thai.HF_REVISION}/{shard}"
    key = adapter.client._key(url)
    body = adapter.client.cache_dir / f"{key}.parquet"
    short = "พระราชบัญญัติ\nพระราชบัญญัติ\nระบบการชำระเงิน\nพ.ศ. ๒๕๖๐\nมาตรา ๑ พระราชบัญญัตินี้เรียกว่า ก"
    long = short + "\nมาตรา ๒ พระราชบัญญัตินี้ใช้บังคับทันที"
    table = pa.table(
        {
            "title": ["x", "พระราชบัญญัติระบบการชำระเงิน พ.ศ. 2560", "อื่น"],
            "text": [short, long, "ประกาศ อื่น"],
            "__index_level_0__": pa.array([7, 9, 11], type=pa.int64()),
        }
    )
    pq.write_table(table, body)
    meta = RawDoc(
        source="thai",
        url=url,
        sha256="2" * 64,
        path=str(body),
        content_type="x",
        fetched_at=datetime.now(UTC),
    )
    (adapter.client.cache_dir / f"{key}.meta.json").write_text(meta.model_dump_json())
    raw = next(adapter.fetch(["PSA2560"]))
    payload = json.loads(adapter.client.read(raw))
    assert payload["row_id"] == 9 and payload["revision"] == thai.HF_REVISION
    assert raw.url.endswith(f"{shard}#__index_level_0__=9") and spec.hf_needle
    doc, units = adapter.parse(raw)
    assert doc.url == raw.url and [u.unit_id for u in units][:2] == [
        "thai/PSA2560/s1/p1",
        "thai/PSA2560/s2/p1",
    ]


@pytest.mark.network
def test_live_fetch_small_english_act(tmp_path: Path) -> None:
    ad = thai.ThaiAdapter(raw_root=tmp_path)
    raw = next(ad.fetch(["FIBA2551-en"]))
    doc, units = ad.parse(raw)
    assert doc.language == "en" and len(units) > 400
    assert "thai/FIBA2551-en/s29/p1" in {u.unit_id for u in units}


def test_broken_sara_am_repaired() -> None:
    from regslm.corpus.thai import _clean

    assert _clean("ให้ธนาคารพาณิชย์น ายอดเงินที่ครบก าหนดช าระ") == "ให้ธนาคารพาณิชย์นำยอดเงินที่ครบกำหนดชำระ"
    assert _clean("ต่าง ๆ ที่") == "ต่าง ๆ ที่"
