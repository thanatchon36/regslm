"""Thai adapter: statutes and Bank of Thailand notifications (Thai) plus an official English set.

Document types
    * Acts (พระราชบัญญัติ): Securities and Exchange Act B.E. 2535 (``SEA2535``), Payment Systems
      Act B.E. 2560 (``PSA2560``), Financial Institutions Business Act B.E. 2551 (``FIBA2551``),
      Bank of Thailand Act B.E. 2485 (``BOTA2485``).
    * Bank of Thailand notifications (ประกาศธนาคารแห่งประเทศไทย): ~20 current texts on capital,
      liquidity, large exposure (single lending limit, related lending), market conduct and IT
      risk, ``doc_id`` = ``BOTN<file id>`` (the 8-digit id of the BoT PDF, e.g. ``BOTN25680181``).
    * English: the BoT's unofficial English translation of FIBA (``FIBA2551-en``, language ``en``).
    * Not obtained: Anti-Money Laundering Act B.E. 2542 (``AMLA2542``), see docs/sources/NEEDED.md.

Access routes (robots.txt honoured through ``PoliteClient``; nothing is evaded)
    The Royal Gazette (ratchakitcha.soc.go.th) and SEC Thailand (sec.or.th) answer automated
    clients with a WAF/Cloudflare block and are not used. krisdika.go.th is unreachable and the
    AMLO attachments sit under ``/amlo-intranet/`` which its robots.txt disallows.

    1. Hugging Face dataset ``pythainlp/thailaw-v1.0`` (Office of the Council of State texts;
       card: ``license: cc0-1.0``, "License Dataset is public domain."). Parquet shards are
       downloaded from ``https://huggingface.co/datasets/pythainlp/thailaw-v1.0/resolve/<sha>/
       data/train-0000{0,1}-of-00002.parquet`` (revision pinned in ``HF_REVISION``). Columns
       ``title``, ``text``, ``__index_level_0__``. Rows are selected by the *head of the text*
       (the dataset titles are unreliable: the full SEA text sits in the row titled
       "... (ฉบับที่ 6) พ.ศ. 2562"). Only SEA2535 (row 3111) and PSA2560 (row 51667) exist in it;
       FIBA, BOT Act and AMLA are absent from every pythainlp/wannaphong thailaw dataset.
       The selected row is stored as a small JSON "extract" next to the cached shard; its
       ``Document.url`` is ``.../blob/<sha>/data/<shard>#__index_level_0__=<row id>``.
    2. www.bot.or.th (robots.txt: only ``/_layouts/``, ``/_vti_bin/``, ``/_catalogs/`` disallowed):
       the BoT law pages host the Council of State's consolidated Thai PDFs of FIBA and the BOT
       Act (``.../legal-department/5-financial-act/5-1-financial-act/5.1.1-Law_TH_FinancialActs
       Updated-2561.pdf``, ``.../1-bot-act/1-1-bot-act/1.1.1-Law_Th_BOT-2561Updated.pdf``), the
       notifications (``/content/dam/bot/fipcs/documents/FPG/<year>/ThaiPDF/<id>.pdf``, linked from
       the "Index of BOT announcements" PDFs) and the English FIBA translation
       (``.../en/laws-and-rules/.../5.1%20LAW05_FIAct.pdf``).

PDF text repair (``pypdf``; no new dependencies)
    * Microsoft Thai private-use glyphs ``U+F701-U+F71A`` (tone marks, ``ั ็ ์`` ...) are mapped
      back to the standard Thai marks, otherwise tone marks vanish ("แหงประเทศไทย").
    * ``ํ`` + ``า`` is joined to ``ำ``. Council of State PDFs encode ``า`` as ``ำ`` and sara am as
      " ำ" (space + ำ); detected per document (``กำร`` >> ``การ``) and reversed. The repeated
      watermark "สำนักงานคณะกรรมการกฤษฎีกา" is dropped.
    * Superscript footnote markers (font size < 70 % of body) become ``[n]`` and footnote lines are
      collected separately (used for ``last_amended``, never emitted as text).
    * Garbled-extraction guard: per page, drop the page if the share of dictionary words
      (``pythainlp.corpus.thai_words`` over ``word_tokenize`` tokens) is < 0.5 or the tone-mark
      sentinel (ให้/ได้ against ให/ได) is < 0.6; a document with > 50 % dropped pages (or fewer than
      two text pages) raises ``GarbledPdfError`` and is listed in ``adapter.dropped``.
    * ``pythainlp.util.normalize`` is applied to all Thai text; Thai digits are kept in ``text``.

Structure and units
    Act: หมวด (chapter) > ส่วน (part) > มาตรา N (section) > วรรค (paragraph, one per text block of
    the section) > (๑) items > (ก) sub-items. Unit = วรรค, or the leaf item when a วรรค has items
    (the lead-ins, joined by `` | ``, go into ``heading`` after the chapter/part labels). Preamble,
    signatures, หมายเหตุ and footnotes are skipped; transitory provisions are kept. Repealed
    ("ยกเลิก") and empty units are skipped.
    ``path`` = ``["ch1", "pt1", "s29", "p2", "i3"]`` (ASCII; ``s29_1`` for มาตรา ๒๙/๑,
    ``s29bis`` for ทวิ); ``unit_id`` = ``thai/FIBA2551/s29/p2/i3`` (no chapter/part, ASCII
    digits). Items are
    ``i<n>`` for numbers and ``k<n>`` for Thai consonant items (ก = ``k1``). English sections use
    the same tokens, so Thai and English units align on the tail of ``path`` starting at ``s..``.
    Notification: numbered clauses ``5.3.1`` (token ``c5.3.1``) > paragraphs ``p<n>`` > items. A
    short numbered clause with body paragraphs below it acts as a heading, otherwise its text is
    paragraph 1. The notification text before the first clause and attachment tables
    (เอกสารแนบ ...) are skipped; the signed cover pages of BoT PDFs are images and carry no text.

``last_amended``
    ``YYYY-01-01`` from the latest B.E. year in a footnote of the section that says it was amended
    ("แก้ไขเพิ่มเติมโดย ... พ.ศ. 2558" gives 2015-01-01); None if there is no such note.
``version``
    HF extracts: the date of the pinned dataset revision. PDFs: ``YYYY-01-01`` of the latest
    amendment year found in the footnotes, else the B.E. year in the BoT file id / the act's
    enactment year.

Licence
    Copyright Act B.E. 2537 s. 7(2)-(3) (acts, regulations, notifications of government agencies
    are not copyright works) and s. 7(5) (translations made by government agencies); the English
    FIBA text is the BoT's own translation (tier P, courtesy line "Unofficial translation; Thai
    text prevails"). Third-party (law-firm) translations are tier L and are not fetched.

Default selection: SEA2535, PSA2560, FIBA2551, BOTA2485, FIBA2551-en and the BoT notifications in
``NOTICES``; ``selection`` takes any of these doc_ids (AMLA2542 raises with a pointer to NEEDED.md).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from io import BytesIO
from itertools import pairwise
from math import hypot
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pypdf
from pydantic import BaseModel

from regslm.corpus.base import (
    Document,
    Licence,
    RawDoc,
    SourceAdapter,
    Unit,
    make_unit_id,
    sha256_bytes,
    write_jsonl,
)

HF_REPO = "pythainlp/thailaw-v1.0"
HF_REVISION = "4c1e2c091b15b5f01d836d5a23398594157e021c"
HF_REVISION_DATE = "2024-02-25"
HF_SHARDS = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
HF_CARD_LICENCE = "License Dataset is public domain."
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}"
BOT = "https://www.bot.or.th/content/dam/bot"
BOT_LAWS_TH = f"{BOT}/documents/th/laws-and-rules/laws-and-regulations/legal-department"
BOT_LAWS_EN = f"{BOT}/documents/en/laws-and-rules/laws-and-regulations/legal-department"

LICENCE = Licence(
    name="Thai Copyright Act B.E. 2537 s. 7(2)-(3),(5): laws, notifications and government "
    "translations are not copyright works",
    url="https://www.ananda-ip.com/files/Copyright-Act.pdf",
    redistributable=True,
    derivative_ok=True,
    attribution="Source: Royal Gazette / Office of the Council of State / Bank of Thailand, "
    "Thailand. English text: unofficial translation; Thai text prevails.",
)

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
TH_NUM = "[๐-๙0-9]"
# Microsoft Thai PUA glyph variants (shifted vowels / tone marks) -> standard Thai marks
PUA_TO_THAI = dict(zip(range(0xF701, 0xF71B), "ิีึื่้๊๋์่้๊๋์ฺัํ็่้๊๋์ฺุู", strict=True))
WATERMARK = "สานักงานคณะกรรมการกฤษฎีกา"


class GarbledPdfError(RuntimeError):
    """Raised when a PDF's text layer is unusable (garbled font mapping / image-only)."""


class ThaiBlockedError(RuntimeError):
    """Raised when a document cannot be fetched without evading a block or robots.txt."""


@dataclass(frozen=True)
class DocSpec:
    doc_id: str
    title: str
    language: str
    kind: str  # act | notice
    route: str  # hf | pdf
    url: str
    topic: str = ""
    enacted: str = ""  # ISO date of the act (version fallback)
    hf_needle: str = ""  # head of the text, whitespace-free, ASCII digits
    parallel_of: str = ""  # for -en documents: the Thai doc_id


def _notice(fid: str, topic: str, title: str, folder: str = "FPG") -> DocSpec:
    return DocSpec(
        doc_id=f"BOTN{fid}",
        title=title,
        language="th",
        kind="notice",
        route="pdf",
        url=f"{BOT}/fipcs/documents/{folder}/{fid[:4]}/ThaiPDF/{fid}.pdf",
        topic=topic,
    )


ACTS: list[DocSpec] = [
    DocSpec(
        "SEA2535",
        "พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์ พ.ศ. ๒๕๓๕",
        "th",
        "act",
        "hf",
        f"{HF_BASE}/blob/{HF_REVISION}/{HF_SHARDS[0]}",
        "securities",
        "1992-03-12",
        hf_needle="พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์พ.ศ.2535",
    ),
    DocSpec(
        "PSA2560",
        "พระราชบัญญัติระบบการชำระเงิน พ.ศ. ๒๕๖๐",
        "th",
        "act",
        "hf",
        f"{HF_BASE}/blob/{HF_REVISION}/{HF_SHARDS[1]}",
        "payments",
        "2017-10-16",
        hf_needle="พระราชบัญญัติระบบการชำระเงินพ.ศ.2560",
    ),
    DocSpec(
        "FIBA2551",
        "พระราชบัญญัติธุรกิจสถาบันการเงิน พ.ศ. ๒๕๕๑",
        "th",
        "act",
        "pdf",
        f"{BOT_LAWS_TH}/5-financial-act/5-1-financial-act/5.1.1-Law_TH_FinancialActsUpdated-2561.pdf",
        "banking",
        "2008-01-27",
    ),
    DocSpec(
        "BOTA2485",
        "พระราชบัญญัติธนาคารแห่งประเทศไทย พุทธศักราช ๒๔๘๕",
        "th",
        "act",
        "pdf",
        f"{BOT_LAWS_TH}/1-bot-act/1-1-bot-act/1.1.1-Law_Th_BOT-2561Updated.pdf",
        "central bank",
        "1942-01-01",
    ),
    DocSpec(
        "FIBA2551-en",
        "Financial Institution Business Act, B.E. 2551 (2008)",
        "en",
        "act",
        "pdf",
        f"{BOT_LAWS_EN}/5-financial-act/5.1%20LAW05_FIAct.pdf",
        "banking",
        "2008-01-27",
        parallel_of="FIBA2551",
    ),
]

NOTICES: list[DocSpec] = [
    _notice("25550332", "capital", "หลักเกณฑ์การกำกับดูแลเงินกองทุนสำหรับธนาคารพาณิชย์"),
    _notice("25620136", "capital", "องค์ประกอบของเงินกองทุนสำหรับธนาคารพาณิชย์ที่จดทะเบียนในประเทศ"),
    _notice("25640144", "capital", "หลักเกณฑ์การกำกับดูแลเงินกองทุนโดยทางการ (Pillar 2)"),
    _notice(
        "25680199",
        "capital",
        "หลักเกณฑ์การกำกับดูแลเงินกองทุนสำหรับธนาคารพาณิชย์และกลุ่มธุรกิจ"
        "ทางการเงินของธนาคารพาณิชย์ที่ลงทุนในธนาคารพาณิชย์ไร้สาขา (Virtual Bank)",
    ),
    _notice("25620139", "capital", "หลักเกณฑ์การกำกับดูแลเงินกองทุนสำหรับกลุ่มธุรกิจทางการเงิน"),
    _notice("25660192", "capital", "หลักเกณฑ์การกำกับดูแลเงินกองทุนสำหรับกลุ่มธุรกิจทางการเงิน"),
    _notice("25510555", "capital", "หลักเกณฑ์การดำรงเงินกองทุนขั้นต่ำสำหรับความเสี่ยงด้านปฏิบัติการ"),
    _notice("25560150", "capital", "การเปิดเผยข้อมูลเกี่ยวกับการดำรงเงินกองทุนสำหรับธนาคารพาณิชย์"),
    _notice("25620141", "capital", "การเปิดเผยข้อมูลการดำรงเงินกองทุนสำหรับธนาคารพาณิชย์"),
    _notice(
        "25580129",
        "liquidity",
        "หลักเกณฑ์การดำรงสินทรัพย์สภาพคล่องเพื่อรองรับสถานการณ์"
        "ด้านสภาพคล่องที่มีความรุนแรง (Liquidity Coverage Ratio: LCR)",
    ),
    _notice(
        "25630082",
        "liquidity",
        "หลักเกณฑ์การดำรงสินทรัพย์สภาพคล่องเพื่อรองรับสถานการณ์ด้านสภาพคล่องที่มีความรุนแรง (LCR) (ฉบับที่ 2)",
    ),
    _notice(
        "25610037",
        "liquidity",
        "หลักเกณฑ์การดำรงแหล่งที่มาของเงินให้สอดคล้องกับการใช้ไปของเงิน (Net Stable Funding Ratio: NSFR)",
    ),
    _notice(
        "25610041",
        "liquidity",
        "การเปิดเผยข้อมูลการดำรงสินทรัพย์สภาพคล่องเพื่อรองรับสถานการณ์"
        "ด้านสภาพคล่องที่มีความรุนแรง (LCR disclosure standards)",
    ),
    _notice("25530017", "liquidity", "แนวนโยบายการบริหารความเสี่ยงด้านสภาพคล่องของสถาบันการเงิน"),
    _notice("25680181", "large exposure", "หลักเกณฑ์การกำกับลูกหนี้รายใหญ่ (Single Lending Limit)"),
    _notice(
        "25680195",
        "large exposure",
        "หลักเกณฑ์การกำกับการทำธุรกรรมกับผู้ถือหุ้นรายใหญ่หรือกิจการที่เกี่ยวข้อง (Related Lending)",
    ),
    _notice(
        "25630223", "market conduct", "การบริหารจัดการด้านการให้บริการแก่ลูกค้าอย่างเป็นธรรม (Market conduct)"
    ),
    _notice(
        "25630225",
        "market conduct",
        "การกำหนดแนวทางการพิจารณานำหลักเกณฑ์การบริหารจัดการ"
        "ด้านการให้บริการแก่ลูกค้าอย่างเป็นธรรม (Market conduct) ไปถือปฏิบัติ",
    ),
    _notice("25680226", "market conduct", "หลักเกณฑ์การกำกับดูแลให้มีกลไกให้ผู้ใช้บริการใช้สิทธิส่งข้อร้องเรียน"),
    _notice("25620138", "capital", "องค์ประกอบของเงินกองทุนสำหรับสาขาของธนาคารพาณิชย์ต่างประเทศ"),
    _notice(
        "25630304", "capital", "องค์ประกอบของเงินกองทุนสำหรับธนาคารพาณิชย์ที่จดทะเบียนในประเทศ (ฉบับที่ 2)"
    ),
    _notice(
        "25620143", "capital", "หลักเกณฑ์การกำกับดูแลด้านเงินกองทุนและการดำรงสินทรัพย์สภาพคล่องสำหรับบริษัทเงินทุน"
    ),
    _notice(
        "25620145",
        "capital",
        "หลักเกณฑ์การกำกับดูแลด้านเงินกองทุนและการดำรงสินทรัพย์สภาพคล่องสำหรับบริษัทเครดิตฟองซิเอร์",
    ),
    _notice(
        "25660202",
        "IT risk",
        "หลักเกณฑ์การกำกับดูแลความเสี่ยงด้านเทคโนโลยีสารสนเทศ "
        "(Information Technology Risk) ของสถาบันการเงินและสถาบันการเงินเฉพาะกิจ",
        "FOG",
    ),
]

UNAVAILABLE = {
    "AMLA2542": "Anti-Money Laundering Act B.E. 2542 is in no pythainlp/wannaphong thailaw "
    "dataset and AMLO hosts it under /amlo-intranet/ (robots.txt disallow); see "
    "docs/sources/NEEDED.md"
}
SPECS: dict[str, DocSpec] = {s.doc_id: s for s in [*ACTS, *NOTICES]}
_BY_URL = {s.url: s for s in SPECS.values()}


# --------------------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------------------
def asc(s: str) -> str:
    return s.translate(THAI_DIGITS)


_C1_FIX = {0x91: "‘", 0x92: "’", 0x93: "“", 0x94: "”", 0x96: "–", 0x97: "—"}


def _normalize(s: str) -> str:
    from pythainlp.util import normalize

    s = s.translate(_C1_FIX).replace("ํา", "ำ").replace("\xa0", " ").replace("​", "")
    return str(normalize(s))


# BoT notification PDFs drop the nikhahit of sara am, leaving "ก าหนด" for "กำหนด". A space
# followed by sara aa is never valid Thai (it cannot start a syllable), so rejoin as sara am.
_BROKEN_SARA_AM = re.compile(r"([ก-ฮ][\u0e48-\u0e4b]?) า")


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", _normalize(s))
    s = _BROKEN_SARA_AM.sub(r"\1ำ", s)
    return re.sub(r"“ ", "“", re.sub(r" ”", "”", s)).strip()


def _is_th(ch: str) -> bool:
    """Thai letter or mark (Thai digits excluded: they separate from letters)."""
    return "\u0e00" <= ch <= "\u0e4f" or "\u0e5a" <= ch <= "\u0e7f"


def _join(a: str, b: str) -> str:
    """Join two wrapped lines: Thai is written without spaces, so only add one at script edges."""
    a, b = a.rstrip(), b.lstrip()
    if not a or not b:
        return a or b
    if _is_th(a[-1]) and _is_th(b[0]) and not b.startswith("พ.ศ."):
        return a + b
    return a + " " + b


_EMPTY_RE = re.compile(
    r"^[\[\(]?\s*(ยกเลิก|reserved|deleted|repealed|omitted|ข้อความเดิม.*)\s*[\]\)]?\.?$", re.I
)
_PAGE_NO_RE = re.compile(r"^[-–\s]*(" + TH_NUM + r"+|[ivx]+)[-–\s]*$")
_LINE_MARKER_RE = re.compile(
    r"^\s*("
    + TH_NUM
    + r"{1,2}(\.[๐-๙0-9]{1,2})+\.?(\s|(?=[฀-๿]))|"
    + TH_NUM
    + r"{1,2}\.(\s|(?=[฀-๿]))|"
    r"\(\s*(" + TH_NUM + r"{1,2}(\.[๐-๙0-9]{1,2})*|[ก-ฮ]{1,2})\s*\)\s|[-•●]\s)"
)
_FN_MARK_RE = re.compile(r"\[(" + TH_NUM + r"{1,3})\]")


def _fix_pua(s: str) -> str:
    return s.translate(PUA_TO_THAI)


def _year_ce(be: str) -> int:
    return int(asc(be)) - 543


def _amend_year(text: str) -> int | None:
    """Latest B.E. year (as CE) of an amending act named in a footnote."""
    if re.search(
        r"(?i)\b(as amended|as added|as repealed|as replaced|amended by|added by)\b", text
    ):
        ces = [int(y) for y in re.findall(r"\((\d{4})\)", text)]
        return max(ces) if ces else None
    if not re.search(r"แก้ไขเพิ่มเติม|เพิ่มเติมโดย|เพิ่มโดย|แก้ไขโดย|ยกเลิกและ|ใช้ความต่อไปนี้แทน|ยกเลิกโดย", text):
        return None
    ys = [_year_ce(y) for y in re.findall(r"พ\.ศ\.\s*(" + TH_NUM + r"{4})", text)]
    ys = [y for y in ys if 1900 < y < 2100]
    return max(ys) if ys else None


# --------------------------------------------------------------------------------------
# PDF -> blocks
# --------------------------------------------------------------------------------------
@dataclass
class Block:
    text: str
    marks: list[str] = field(default_factory=list)
    page: int = 0


@dataclass
class Extract:
    blocks: list[Block]
    footnotes: dict[str, str]
    pages_total: int
    pages_dropped: list[int]


@dataclass
class _Line:
    x: float
    text: str
    fn: bool
    marks: list[str]
    y: float


@dataclass
class _Chunk:
    x: float
    y: float
    size: float
    text: str


def _chunks(page: pypdf.PageObject) -> list[_Chunk]:
    out: list[_Chunk] = []

    def visitor(text: str, cm: list[float], tm: list[float], _fd: Any, fs: float) -> None:
        if not text:
            return
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        size = (fs or 1.0) * hypot(tm[2], tm[3]) * hypot(cm[2], cm[3])
        out.append(_Chunk(x, y, size, text))

    page.extract_text(visitor_text=visitor)
    return out


def _is_watermark(text: str) -> bool:
    t = re.sub(r"[\sำา]", "", text)
    return bool(t) and len(t) >= 2 and t in WATERMARK.replace("า", "") * 3


def _cluster(values: list[float], tol: float) -> list[list[int]]:
    """Group indexes of ``values`` whose neighbours differ by at most ``tol``."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and abs(values[groups[-1][-1]] - values[i]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _join_chunks(parts: list[tuple[float, str]], lang: str) -> str:
    """Concatenate the text chunks of one line; for Latin text add a space where there is a gap."""
    if lang != "en" or len(parts) < 2:
        return "".join(p[1] for p in parts)
    chars = sum(len(p[1]) for p in parts[:-1])
    w = (parts[-1][0] - parts[0][0]) / chars if chars else 0.0
    out = [parts[0][1]]
    for (x, t), (nx, nt) in pairwise(parts):
        gap = nx - (x + len(t) * w)
        if w > 0 and gap > 0.35 * w and not t.endswith(" ") and not nt.startswith(" "):
            out.append(" ")
        out.append(nt)
    return "".join(out)


def _dictionary_cover(text: str) -> float:
    from pythainlp.corpus.common import thai_words
    from pythainlp.tokenize import word_tokenize

    words = thai_words()
    toks = [t for t in word_tokenize(text[:40000], engine="newmm") if re.search("[ก-ฮ]", t)]
    tot = sum(len(t) for t in toks)
    return sum(len(t) for t in toks if t in words) / tot if tot else 1.0


def _tone_sentinel(text: str) -> tuple[int, float]:
    good = len(re.findall("ให้|ได้", text))
    bad = len(re.findall("ให(?![้ญ])|ได(?!้)", text))
    return good + bad, good / (good + bad) if good + bad else 1.0


def page_is_garbled(text: str) -> bool:
    thai = sum(_is_th(c) for c in text)
    if len(text) > 300 and thai < 0.1 * len(text.replace(" ", "")):
        return True  # almost no Thai in a Thai document: glyph-name junk
    if thai < 40:
        return False  # tables / latin pages are judged elsewhere
    if _dictionary_cover(text) < 0.5:
        return True
    n, ratio = _tone_sentinel(text)
    return n >= 6 and ratio < 0.6


def extract_blocks(data: bytes, lang: str, footnotes_on: bool = True) -> Extract:
    """Read a PDF into paragraph blocks (first-line indentation marks a new paragraph)."""
    reader = pypdf.PdfReader(BytesIO(data))
    raw_pages = [_chunks(p) for p in reader.pages]
    for chunks in raw_pages:
        for c in chunks:
            c.text = _fix_pua(c.text)
    sizes: Counter[float] = Counter()
    sample = []
    for chunks in raw_pages:
        for c in chunks:
            if c.text.strip() and not _is_watermark(c.text):
                sizes[round(c.size * 2) / 2] += len(c.text.strip())
                sample.append(c.text)
    if not sizes:
        raise GarbledPdfError("no text layer")
    body = sizes.most_common(1)[0][0]
    joined = "".join(sample)
    swap = lang == "th" and joined.count("กำร") > 3 * joined.count("การ")

    def fix(s: str) -> str:
        s = s.replace("ํา", "ำ")
        if swap:
            s = re.sub(r"(?<=\S) ำ", "\x00", s).replace("ำ", "า").replace("\x00", "ำ")
        return s

    parsed_pages: list[tuple[int, list[_Line]]] = []
    dropped: list[int] = []
    for pno, chunks in enumerate(raw_pages):
        chunks = [
            c
            for c in chunks
            if c.text.strip()
            and not _is_watermark(c.text)
            and not (c.size < 0.9 * body and re.fullmatch(r"[\sกคงตนรมสฤษฎีัำาะกณ]{1,3}", c.text))
        ]
        bodyc = [c for c in chunks if c.size >= (0.9 * body if footnotes_on else 0.7 * body)]
        foot_lo = 0.7 * body if footnotes_on else 0.0
        footc = [c for c in chunks if foot_lo <= c.size < 0.9 * body] if footnotes_on else []
        smallc = [c for c in chunks if c.size < 0.7 * body]
        lines: list[tuple[float, list[_Chunk], bool]] = []  # (y, chunks, is_footnote)
        for is_fn, group in ((False, bodyc), (True, footc)):
            for idxs in _cluster([c.y for c in group], 2.5):
                cs = sorted((group[i] for i in idxs), key=lambda c: c.x)
                lines.append((statistics.fmean(c.y for c in cs), cs, is_fn))
        lines.sort(key=lambda t: -t[0])
        marks_for: dict[int, list[tuple[float, str]]] = {}
        for c in smallc:
            if not lines or not re.search(r"\w", c.text):
                continue
            j = min(range(len(lines)), key=lambda k: abs(lines[k][0] - c.y))
            if abs(lines[j][0] - c.y) <= 9:
                marks_for.setdefault(j, []).append((c.x, c.text.strip()))
        page_lines: list[_Line] = []
        for j, (_y, cs, is_fn) in enumerate(lines):
            parts: list[tuple[float, str]] = [(c.x, c.text) for c in cs]
            smalls = sorted(marks_for.get(j, []))
            fn_marker: list[str] = []
            if is_fn:
                if smalls and smalls[0][0] <= min(c.x for c in cs):
                    fn_marker = [smalls.pop(0)[1]]
                parts += [(x, m) for x, m in smalls]
            else:
                parts += [(x, f" [{m}] ") for x, m in smalls]
            parts.sort(key=lambda t: t[0])
            txt = re.sub(r"\s+([,;:.)])", r"\1", fix(_join_chunks(parts, lang)))
            page_lines.append(_Line(min(c.x for c in cs), txt, is_fn, fn_marker, _y))
        body_text = "\n".join(ln.text for ln in page_lines if not ln.fn)
        if lang == "th" and page_is_garbled(body_text):
            dropped.append(pno)
            continue
        parsed_pages.append((pno, page_lines))
    mins = Counter(
        round(min(ln.x for ln in pl if not ln.fn) / 2) * 2
        for _p, pl in parsed_pages
        if any(not ln.fn for ln in pl)
    )
    gaps = [
        a.y - b.y
        for _p, pl in parsed_pages
        for a, b in pairwise([ln for ln in pl if not ln.fn])
        if 0 < a.y - b.y < 40
    ]
    line_gap = statistics.median(gaps) if gaps else 0.0
    lens = sorted(len(ln.text) for _p, pl in parsed_pages for ln in pl if not ln.fn)
    full_len = lens[int(0.9 * (len(lens) - 1))] if lens else 0
    margin = mins.most_common(1)[0][0] if mins else 0
    blocks: list[Block] = []
    footnotes: dict[str, str] = {}
    for pno, page_lines in parsed_pages:
        cur_fn: str | None = None
        cur: Block | None = None
        prev_y: float | None = None
        prev_short = False
        for ln in page_lines:
            x0, txt, fn, fmark = ln.x, ln.text, ln.fn, ln.marks
            if fn:
                if fmark:
                    cur_fn = asc(fmark[0])
                    footnotes[cur_fn] = txt.strip()
                elif cur_fn:
                    footnotes[cur_fn] = _join(footnotes[cur_fn], txt)
                continue
            cur_fn = None
            if not txt.strip() or _PAGE_NO_RE.match(txt):
                continue
            if footnotes_on:  # acts: first-line indentation starts a paragraph
                new = cur is None or x0 - margin >= 6
            else:  # notices: numbered markers or a larger vertical gap start a block
                spaced = prev_y is not None and prev_y - ln.y > 1.35 * line_gap
                new = cur is None or spaced or prev_short or bool(_LINE_MARKER_RE.match(txt))
            prev_short = (
                not footnotes_on
                and new
                and bool(_LINE_MARKER_RE.match(txt))
                and len(txt) < 0.7 * full_len
            )
            prev_y = ln.y
            if new:
                cur = Block(txt.strip(), page=pno)
                blocks.append(cur)
            else:
                assert cur is not None
                cur.text = _join(cur.text, txt)
    text_pages = sum(1 for _p, pl in parsed_pages if any(not ln.fn for ln in pl)) + len(dropped)
    if text_pages < max(1.0, min(2.0, 0.2 * len(raw_pages))):
        raise GarbledPdfError(
            f"image-only PDF: {text_pages}/{len(raw_pages)} pages have a text layer"
        )
    if len(dropped) > 0.5 * text_pages:
        raise GarbledPdfError(f"{len(dropped)}/{text_pages} text pages garbled")
    for b in blocks:
        b.text = _clean(b.text)
        b.marks = [asc(m) for m in _FN_MARK_RE.findall(b.text)]
        b.text = _clean(_FN_MARK_RE.sub("", b.text))
    page_no = _PAGE_NO_RE
    lead_no = re.compile(r"^[-–]\s*" + TH_NUM + r"+\s*[-–]\s*")
    for b in blocks:
        b.text = lead_no.sub("", b.text)
    blocks = [b for b in blocks if b.text and not page_no.match(b.text)]
    footnotes = {k: _clean(v) for k, v in footnotes.items()}
    return Extract(blocks, footnotes, len(raw_pages), dropped)


def blocks_from_text(text: str) -> Extract:
    """Blocks for a plain-text statute (HF rows): ``\\n`` ends a paragraph, ``\\r\\n`` wraps."""
    text = _fix_pua(text).replace("ํา", "ำ").replace("\r\n", "\r")
    blocks: list[Block] = []
    footnotes: dict[str, str] = {}
    for para in text.split("\n"):
        para = re.sub(r"\s*\r\s*", "\r", para)
        parts = [p for p in para.split("\r") if p.strip()]
        if not parts:
            continue
        s = parts[0]
        for p in parts[1:]:
            s = _join(s, p)
        s = _clean(s)
        if not s:
            continue
        m = re.match(r"^\[(" + TH_NUM + r"{1,3})\]\s*(.+)$", s)
        if m:
            footnotes[asc(m.group(1))] = m.group(2)
            continue
        marks = [asc(x) for x in _FN_MARK_RE.findall(s)]
        s = _clean(_FN_MARK_RE.sub("", s))
        if s:
            blocks.append(Block(s, marks))
    return Extract(blocks, footnotes, 0, [])


# --------------------------------------------------------------------------------------
# segmentation of blocks into units
# --------------------------------------------------------------------------------------
_TH_SUFFIX = {"ทวิ": "bis", "ตรี": "ter", "จัตวา": "quater", "เบญจ": "quinquies"}
_TH_DIV_RE = re.compile(r"^(หมวด|ส่วนที่|ส่วน)\s*(" + TH_NUM + r"+(?:/" + TH_NUM + r"+)?)\s*(.*)$")
_TH_TRANS_RE = re.compile(r"^(บทเฉพาะกาล|บททั่วไป|บทเบ็ดเตล็ด)\s*(.*)$")
_TH_SEC_RE = re.compile(
    r"^มาตรา\s*(" + TH_NUM + r"+)(?:\s*/\s*(" + TH_NUM + r"+))?(?:\s*(ทวิ|ตรี|จัตวา|เบญจ))?\s+(.*)$"
)
_TH_SEC_ONLY_RE = re.compile(r"^มาตรา\s*(" + TH_NUM + r"+)(?:\s*/\s*(" + TH_NUM + r"+))?$")
_EN_DIV_RE = re.compile(r"^(Chapter|Part|Title)\s+(\d+(?:/\d+)?|[IVXL]+)\b\s*(.*)$", re.I)
_EN_SEC_RE = re.compile(
    r"^Section\s+(\d+)(?:\s*/\s*(\d+))?(?:\s+(bis|ter|quater|quinquies))?\.?\s*(?:\[(\d+)\]\s*)?(.*)$"
)
_ITEM_RE = re.compile(
    r"^\(\s*([๐-๙]+(?:\.[๐-๙]+)*|\d+(?:\.\d+)*|[ก-ฮ]{1,2}|[a-z]{1,4})\s*\)\s*(.*)$"
)
_BULLET_RE = re.compile(r"^[-–•●]\s+(.*)$")
_CLAUSE_RE = re.compile(
    r"^(" + TH_NUM + r"{1,2}(?:\." + TH_NUM + r"{1,2}){0,4})\.?(?:\s+|(?=[\u0e00-\u0e7f]))(\S.*)$"
)
_STOP_ACT_RE = re.compile(r"^(ผู้รับสนองพระบรมราชโองการ|หมายเหตุ|Countersigned|Remarks?:|Note:)")
_STOP_NOTICE_RE = re.compile(r"(?!)")  # notices have no stop marker; attachments get a prefix
_ATTACH_RE = re.compile(r"^(?:เอกสารแนบ|ภาคผนวก)\s*(" + TH_NUM + r"+)\s*$")
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}


@dataclass
class _Node:
    token: str
    text: str
    parent: _Node | None
    kind: str  # para | digit | letter | roman | bullet
    value: str = ""
    children: int = 0
    marks: list[str] = field(default_factory=list)


_TH_LETTERS = "กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"


def _item_kind(val: str, lang: str, prev_letter: str | None) -> tuple[str, str]:
    v = asc(val)
    if re.fullmatch(r"\d+(\.\d+)+", v):
        return "dotted", v.replace(".", "_")
    if v.isdigit():
        return "digit", str(int(v))
    if lang == "th":
        return "letter", "".join(str(_TH_LETTERS.find(c) + 1) for c in val)
    if v in _ROMAN and (len(v) > 1 or prev_letter != "h"):
        return "roman", str(sorted(_ROMAN, key=lambda x: (len(x), x)).index(v) + 1)
    return "letter", str(ord(v[0]) - 96)


class _Segmenter:
    def __init__(
        self, spec: DocSpec, footnotes: dict[str, str], version: str, stop_re: re.Pattern[str]
    ) -> None:
        self.spec = spec
        self.lang = spec.language
        self.footnotes = footnotes
        self.version = version
        self.stop_re = stop_re
        self.units: list[Unit] = []
        self.skipped = Counter[str]()
        self.divs: list[tuple[str, str, str]] = []  # (kind, token, display)
        self._seen_ids: set[str] = set()
        self._nodes: list[_Node] = []
        self._section: tuple[str] | None = None
        self._last_key = (0, 0, 0)
        self._para = 0
        self._stack: list[_Node] = []
        self._sec_marks: list[str] = []
        self._div_path: list[str] = []

    # ---- emission -------------------------------------------------------------------
    def _flush(self) -> None:
        if self._section is None:
            self._nodes, self._stack = [], []
            return
        sec_tok = self._section[0]
        years = [y for m in self._sec_marks if (y := _amend_year(self.footnotes.get(m, "")))]
        for n in self._nodes:
            years += [y for m in n.marks if (y := _amend_year(self.footnotes.get(m, "")))]
        last = f"{max(years)}-01-01" if years else None
        heading_base = " > ".join(d[2] for d in self.divs)
        for n in self._nodes:
            if n.children:
                continue
            text = _clean(n.text)
            letters = sum(c.isalpha() or _is_th(c) for c in text)
            if letters < 0.3 * len(text) or letters < 4 or _EMPTY_RE.match(text):
                self.skipped["empty"] += 1
                continue
            chain: list[_Node] = []
            p: _Node | None = n.parent
            while p is not None:
                chain.append(p)
                p = p.parent
            chain.reverse()
            tokens = [c.token for c in chain] + [n.token]
            lead = [c.text for c in chain if c.kind != "para" or c.text]
            heading = " | ".join(x for x in [heading_base, *lead] if x)
            uid = make_unit_id("thai", self.spec.doc_id, *self._id_prefix(sec_tok), *tokens)
            base, c = uid, 2
            while uid in self._seen_ids:
                uid, c = f"{base}~{c}", c + 1
            self._seen_ids.add(uid)
            path = [*self._div_path, sec_tok, *tokens]
            self.units.append(
                Unit(
                    unit_id=uid,
                    source="thai",
                    doc_id=self.spec.doc_id,
                    path=path,
                    heading=heading,
                    text=text,
                    language=self.lang,
                    version=self.version,
                    last_amended=last,
                )
            )
        self._nodes, self._stack, self._sec_marks = [], [], []

    def _id_prefix(self, sec_tok: str) -> list[str]:
        return [sec_tok]

    # ---- node helpers ---------------------------------------------------------------
    def _new_para(self, text: str, marks: list[str]) -> None:
        self._para += 1
        n = _Node(f"p{self._para}", text, None, "para", marks=marks)
        self._nodes.append(n)
        self._stack = [n]

    def _new_item(self, val: str, text: str, marks: list[str], bullet: bool = False) -> None:
        if not self._stack:
            self._new_para("", [])
        prev_letter = next((s.value for s in reversed(self._stack) if s.kind == "letter"), None)
        if bullet:
            kind, v = "bullet", str(self._bullets() + 1)
        else:
            kind, v = _item_kind(val, self.lang, prev_letter)
        same = [i for i in range(1, len(self._stack)) if self._stack[i].kind == kind]
        j = same[-1] if same else None
        if j is not None and not (v == "1" and j < len(self._stack) - 1):
            self._stack = self._stack[:j]  # sibling of the deepest item of the same kind
        parent = self._stack[-1]
        tok = {"digit": "i", "letter": "k", "roman": "r", "bullet": "b", "dotted": "d"}[kind] + v
        parent.children += 1
        n = _Node(tok, text, parent, kind, v, marks=marks)
        self._nodes.append(n)
        self._stack = [*self._stack, n]

    def _bullets(self) -> int:
        top = self._stack[-1] if self._stack else None
        return sum(1 for n in self._nodes if n.kind == "bullet" and n.parent is top)

    # ---- driver ---------------------------------------------------------------------
    def run(self, blocks: list[Block]) -> None:
        raise NotImplementedError


class _ActSegmenter(_Segmenter):
    def run(self, blocks: list[Block]) -> None:
        started = False
        i = 0
        while i < len(blocks):
            b = blocks[i]
            t = b.text
            i += 1
            if started and self.stop_re.match(t):
                break
            sec = self._match_section(t)
            if sec is not None:
                key, tok, rest = sec
                if (not started and key == (1, 0, 0)) or (started and key > self._last_key):
                    started = True
                    self._flush()
                    self._last_key = key
                    self._section = (tok,)
                    self._div_path = [d[1] for d in self.divs]
                    self._para = 0
                    self._sec_marks = list(b.marks)
                    if rest:
                        self._add_text(rest, b.marks)
                    continue
            if not started:
                continue
            used = self._match_div(t, blocks, i)
            if used is not None:
                i += used
                continue
            self._add_text(t, b.marks)
        self._flush()

    # section / division matching ---------------------------------------------------
    def _match_section(self, t: str) -> tuple[tuple[int, int, int], str, str] | None:
        if self.lang == "th":
            m = _TH_SEC_RE.match(t)
            if m:
                n, sub, suf, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            else:
                m2 = _TH_SEC_ONLY_RE.match(t)
                if not m2:
                    return None
                n, sub, suf, rest = m2.group(1), m2.group(2), None, ""
        else:
            m3 = _EN_SEC_RE.match(t)
            if not m3:
                return None
            n, sub, suf, rest = m3.group(1), m3.group(2), m3.group(3), m3.group(5)
        num, subn = int(asc(n)), (int(asc(sub)) if sub else 0)
        eng = _TH_SUFFIX.get(suf or "", suf or "")
        rank = list(_TH_SUFFIX.values()).index(eng) + 1 if eng in _TH_SUFFIX.values() else 0
        tok = f"s{num}" + (f"_{subn}" if sub else "") + eng
        return (num, subn, rank), tok, rest.strip()

    def _match_div(self, t: str, blocks: list[Block], i: int) -> int | None:
        """Handle chapter/part headings; returns the number of extra blocks consumed."""
        if self.lang == "th":
            m = _TH_DIV_RE.match(t)
            if m:
                kind = "ch" if m.group(1) == "หมวด" else "pt"
                num, title, disp = (
                    asc(m.group(2)).replace("/", "_"),
                    m.group(3),
                    f"{m.group(1)} {m.group(2)}",
                )
            else:
                m2 = _TH_TRANS_RE.match(t)
                if not m2 or len(t) > 60:
                    return None
                kind, num, title, disp = "ch", "tr", "", t
        else:
            m3 = _EN_DIV_RE.match(t)
            if not m3:
                return None
            kind = "ch" if m3.group(1).lower() in ("chapter", "title") else "pt"
            num, title, disp = (
                m3.group(2).replace("/", "_"),
                m3.group(3),
                f"{m3.group(1)} {m3.group(2)}",
            )
        used = 0
        if not title and i < len(blocks) and num != "tr":
            nxt = blocks[i].text
            if len(nxt) < 140 and not self._starts_marker(nxt):
                title, used = nxt, 1
        self._flush()
        self._section = None
        label = f"{disp} {title}".strip()
        if kind == "ch":
            self.divs = [(kind, f"ch{num}", label)]
        else:
            self.divs = [d for d in self.divs if d[0] == "ch"] + [(kind, f"pt{num}", label)]
        return used

    def _starts_marker(self, t: str) -> bool:
        return bool(
            _TH_SEC_RE.match(t) or _TH_DIV_RE.match(t) or _EN_SEC_RE.match(t) or _EN_DIV_RE.match(t)
        )

    def _add_text(self, text: str, marks: list[str]) -> None:
        if self._section is None:
            return
        m = _ITEM_RE.match(text)
        b = _BULLET_RE.match(text)
        if m:
            self._new_item(m.group(1), m.group(2), marks)
        elif b:
            self._new_item("-", b.group(1), marks, bullet=True)
        else:
            self._new_para(text, marks)


class _NoticeSegmenter(_Segmenter):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self._seen_clauses: set[str] = set()
        self._attach = ""
        self._last_top = 0
        self._titles: list[tuple[str, str]] = []

    def run(self, blocks: list[Block]) -> None:
        for b in blocks:
            t = b.text
            if self._section is not None and self.stop_re.match(t):
                break
            if re.fullmatch(r"[\s\-–]*" + TH_NUM + r"+[\s\-–]*", t):
                continue  # page number
            am = _ATTACH_RE.match(t)
            if am:
                self._flush()
                self._section = None
                self._attach = f"a{asc(am.group(1))}_"
                self._seen_clauses, self._last_top, self._titles = set(), 0, []
                continue
            c = self._clause(t)
            if c is not None:
                self._flush()
                self._open_clause(c[0], c[1], b.marks)
                continue
            if self._section is None:
                continue
            m = _ITEM_RE.match(t)
            bl = _BULLET_RE.match(t)
            if m:
                self._new_item(m.group(1), m.group(2), b.marks)
            elif bl:
                self._new_item("-", bl.group(1), b.marks, bullet=True)
            else:
                self._new_para(t, b.marks)
        self._flush()

    def _clause(self, t: str) -> tuple[str, str] | None:
        m = _CLAUSE_RE.match(t)
        if not m:
            return None
        cid = asc(m.group(1))
        parts = cid.split(".")
        if len(parts) == 1:
            n = int(parts[0])
            first = self._last_top == 0 and n <= 9
            if not first and n not in (self._last_top, self._last_top + 1):
                return None
        elif ".".join(parts[:-1]) not in self._seen_clauses:
            return None
        return cid, m.group(2)

    def _open_clause(self, cid: str, rest: str, marks: list[str]) -> None:
        parts = cid.split(".")
        if len(parts) == 1:
            self._last_top = int(parts[0])
        self._seen_clauses.add(cid)
        self._titles = [(k, v) for k, v in self._titles if cid.startswith(k + ".")]
        self._section = (f"{self._attach}c{cid}",)
        self._div_path = []
        self._para = 0
        self._sec_marks = list(marks)
        # a short clause is the heading of the paragraphs below it; a long one is paragraph 1
        if len(rest) <= 90 and not rest.endswith((":", ";")) and not _ITEM_RE.match(rest):
            self._titles.append((cid, f"{cid} {rest}"))
        else:
            m = _ITEM_RE.match(rest)
            if m:
                self._new_item(m.group(1), m.group(2), marks)
            else:
                self._new_para(rest, marks)
        self.divs = [("t", "", " > ".join(v for _k, v in self._titles))] if self._titles else []


# --------------------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------------------
def _head_key(text: str) -> str:
    return asc(re.sub(r"\s+", "", _fix_pua(text[:300]).replace("ํา", "ำ")))


class ThaiAdapter(SourceAdapter):
    name = "thai"
    licence = LICENCE

    def __init__(self, raw_root: Path = Path("data/raw")) -> None:
        super().__init__(raw_root)
        self.dropped: list[tuple[str, str]] = []
        self.last_extract: Extract | None = None

    # ---- raw storage ----------------------------------------------------------------
    def _store(self, url: str, body: bytes, content_type: str, suffix: str) -> RawDoc:
        key = self.client._key(url)
        path = self.client.cache_dir / f"{key}{suffix}"
        meta = self.client.cache_dir / f"{key}.meta.json"
        if path.exists() and meta.exists():
            return RawDoc.model_validate_json(meta.read_text())
        path.write_bytes(body)
        raw = RawDoc(
            source=self.name,
            url=url,
            sha256=sha256_bytes(body),
            path=str(path),
            content_type=content_type,
            fetched_at=datetime.now(UTC),
        )
        meta.write_text(raw.model_dump_json())
        return raw

    def _get(self, url: str, suffix: str) -> RawDoc:
        try:
            return self.client.get(url, suffix)
        except PermissionError as e:
            raise ThaiBlockedError(f"{e}; see docs/sources/NEEDED.md") from e

    def _hf_extract(self, spec: DocSpec) -> RawDoc:
        shard = next(s for s in HF_SHARDS if spec.url.endswith(s))
        shard_url = f"{HF_BASE}/resolve/{HF_REVISION}/{shard}"
        raw = self._get(shard_url, ".parquet")
        table = pq.read_table(raw.path)
        best: dict[str, Any] | None = None
        for row in table.to_pylist():
            if spec.hf_needle in _head_key(row["text"])[:120] and (
                best is None or len(row["text"]) > len(best["text"])
            ):
                best = row
        if best is None:
            raise RuntimeError(f"{spec.doc_id}: no row starting with {spec.hf_needle!r} in {shard}")
        row_id = int(best["__index_level_0__"])
        payload = {
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "file": shard,
            "row_id": row_id,
            "title": best["title"],
            "text": best["text"],
            "shard_sha256": raw.sha256,
        }
        url = f"{HF_BASE}/blob/{HF_REVISION}/{shard}#__index_level_0__={row_id}"
        return self._store(
            url, json.dumps(payload, ensure_ascii=False).encode(), "application/json", ".json"
        )

    # ---- SourceAdapter --------------------------------------------------------------
    def default_selection(self) -> list[str]:
        return [s.doc_id for s in [*ACTS, *NOTICES]]

    def fetch(self, selection: list[str] | None = None) -> Iterator[RawDoc]:
        for doc_id in selection or self.default_selection():
            if doc_id in UNAVAILABLE:
                raise RuntimeError(UNAVAILABLE[doc_id])
            if doc_id not in SPECS:
                raise KeyError(f"unknown Thai document {doc_id!r}; choose from {list(SPECS)}")
            spec = SPECS[doc_id]
            yield self._hf_extract(spec) if spec.route == "hf" else self._get(spec.url, ".pdf")

    def _spec_for(self, raw: RawDoc) -> DocSpec:
        if raw.url in _BY_URL:
            return _BY_URL[raw.url]
        for spec in SPECS.values():  # HF extract urls carry a row fragment
            if spec.route == "hf" and raw.url.split("#")[0].endswith(spec.url.split("/blob/")[-1]):
                return spec
        raise ValueError(f"not a known Thai source url: {raw.url}")

    def parse(self, raw: RawDoc) -> tuple[Document, list[Unit]]:
        spec = self._spec_for(raw)
        data = self.client.read(raw)
        if spec.route == "hf":
            payload = json.loads(data)
            ext = blocks_from_text(payload["text"])
            doc_url = raw.url
            version = HF_REVISION_DATE
        else:
            ext = extract_blocks(data, spec.language, spec.kind == "act")
            doc_url = spec.url
            years = [y for t in ext.footnotes.values() if (y := _amend_year(t))]
            if years:
                version = f"{max(years)}-01-01"
            elif spec.kind == "notice":
                version = f"{int(spec.doc_id[4:8]) - 543}-01-01"
            else:
                version = spec.enacted
            date.fromisoformat(version)
        self.last_extract = ext
        stop = _STOP_ACT_RE if spec.kind == "act" else _STOP_NOTICE_RE
        seg: _Segmenter = (
            _ActSegmenter(spec, ext.footnotes, version, stop)
            if spec.kind == "act"
            else _NoticeSegmenter(spec, ext.footnotes, version, stop)
        )
        seg.run(ext.blocks)
        if not seg.units:
            raise GarbledPdfError(f"{spec.doc_id}: no units parsed (no usable text structure)")
        doc = Document(
            source=self.name,
            doc_id=spec.doc_id,
            title=spec.title,
            language=spec.language,
            url=doc_url,
            version=version,
            sha256=raw.sha256,
        )
        return doc, seg.units

    # ---- convenience ----------------------------------------------------------------
    def run(
        self, selection: list[str] | None = None
    ) -> Iterator[tuple[Document, list[Unit], Extract | None]]:
        for raw in self.fetch(selection):
            try:
                doc, units = self.parse(raw)
            except GarbledPdfError as e:
                spec = self._spec_for(raw)
                self.dropped.append((spec.doc_id, str(e)))
                continue
            yield doc, units, self.last_extract


def align_parallel(th_units: list[Unit], en_units: list[Unit]) -> list[tuple[str, str]]:
    """Pairs (th_unit_id, en_unit_id) whose path tails starting at the section token match."""

    def key(u: Unit) -> tuple[str, ...] | None:
        for i, p in enumerate(u.path):
            if re.fullmatch(r"s\d+(_\d+)?[a-z]*", p):
                return tuple(u.path[i:])
        return None

    en = {k: u.unit_id for u in en_units if (k := key(u))}
    pairs = []
    for u in th_units:
        k = key(u)
        if k and k in en:
            pairs.append((u.unit_id, en[k]))
    return pairs


def _words(text: str) -> int:
    from pythainlp.tokenize import word_tokenize

    return len([w for w in word_tokenize(text, engine="newmm") if w.strip()])


def main(argv: list[str] | None = None) -> None:
    """Fetch + parse the default selection, write data/local/units/thai*.jsonl, print stats."""
    import random

    argv = sys.argv[1:] if argv is None else argv
    adapter = ThaiAdapter()
    all_units: list[Unit] = []
    by_doc: dict[str, list[Unit]] = {}
    for doc, units, ext in adapter.run(argv or None):
        by_doc[doc.doc_id] = units
        all_units += units
        lens = sorted(len(u.text) for u in units)
        wl = [_words(u.text) for u in units]
        print(
            f"{doc.doc_id:14s} {doc.language} units={len(units):5d} "
            f"chars med={statistics.median(lens):.0f} p95={lens[int(0.95 * (len(lens) - 1))]} "
            f"words med={statistics.median(wl):.0f} p95={sorted(wl)[int(0.95 * (len(wl) - 1))]} "
            f"version={doc.version} pages_dropped={len(ext.pages_dropped) if ext else 0}"
        )
        for u in random.Random(0).sample(units, min(5, len(units))):
            print(f"    {u.unit_id}  {u.text[:120]}")
    for d, why in adapter.dropped:
        print(f"DROPPED {d}: {why}")
    n = write_jsonl(Path("data/local/units/thai.jsonl"), list[BaseModel](all_units))
    pairs: list[tuple[str, str]] = []
    if "FIBA2551" in by_doc and "FIBA2551-en" in by_doc:
        pairs = align_parallel(by_doc["FIBA2551"], by_doc["FIBA2551-en"])
    out = Path("data/local/units/thai_parallel.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for th, en in pairs:
            f.write(json.dumps({"th_unit_id": th, "en_unit_id": en}, ensure_ascii=False) + "\n")
    print(f"wrote {n} units, {len(pairs)} parallel pairs")


if __name__ == "__main__":
    main()
