"""Turn an arbitrary document (PDF / HTML / text / Markdown) into corpus ``Unit`` rows.

Unit ids are ``doc/<n>`` or ``doc/<heading-path>/<n>`` when headings or numbering such as
"Article 5", "§ 3.10", "มาตรา ๕", "ข้อ ๓" are detected. A paragraph ending in ":" becomes the
lead-in (``heading``) of the list items that follow it, as in the corpus adapters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from regslm.corpus.base import Unit

SOURCE = "doc"
SUFFIXES = {".pdf", ".html", ".htm", ".txt", ".md"}
THAI_SHARE = 0.30
_TH_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_NUM = r"[0-9๐-๙]+(?:[.\-/][0-9๐-๙]+)*"
# (pattern, rank): a lower rank is a bigger division and resets deeper levels.
_HEADINGS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(rf"^(Chapter|Part|Title|Division)\s+({_NUM}|[IVXL]+)\b", re.I), 0),
    (re.compile(rf"^(หมวด|ส่วนที่|ภาค|บทที่)\s*({_NUM})"), 0),
    (re.compile(rf"^(Article|Section|Rule|Regulation|Clause)\s+({_NUM})\b", re.I), 1),
    (re.compile(rf"^(§§?)\s*({_NUM})"), 1),
    (re.compile(rf"^(มาตรา|ข้อ)\s*({_NUM})"), 1),
]
_ITEM_RE = re.compile(r"^(\(\s*([0-9๐-๙]+|[a-zA-Zก-ฮ]{1,4})\s*\)|[-–•●*]\s|[0-9๐-๙]+[.)]\s)")
_PAGE_NO_RE = re.compile(r"^[-–\s]*([0-9๐-๙]+|[ivx]+)[-–\s]*$", re.I)
_MAX_HEADING_LEN = 100
_HEADING_ONLY_LEN = 60  # a shorter, unpunctuated heading line carries no obligation
MAX_UNIT_CHARS = 2000  # longer paragraphs are split so prompt + answer fit a 4k context
_SENT_END = re.compile(r"(?<=[.;:!?])\s+")


def thai_share(text: str) -> float:
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return 0.0
    return sum("฀" <= c <= "๿" for c in letters) / len(letters)


def detect_language(text: str, language: str = "auto") -> str:
    """``th`` if more than 30 % of non-space characters are Thai, else ``en``."""
    if language != "auto":
        return language
    return "th" if thai_share(text) > THAI_SHARE else "en"


# ---------------------------------------------------------------- readers -> paragraphs


def _pdf_paragraphs(data: bytes, language: str) -> tuple[list[str], str]:
    import pypdf

    from regslm.corpus.thai import extract_blocks

    lang = language
    if lang == "auto":
        sample = "".join((p.extract_text() or "") for p in pypdf.PdfReader(_bytes_io(data)).pages)
        lang = detect_language(sample)
    paras = [b.text for b in extract_blocks(data, lang).blocks if b.text.strip()]
    return paras, lang


def _bytes_io(data: bytes):
    from io import BytesIO

    return BytesIO(data)


_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "div", "td", "th", "pre",
               "blockquote", "dt", "dd", "caption", "tr"}  # fmt: skip
_SKIP_TAGS = {"script", "style", "head", "noscript", "nav", "footer"}


def _html_paragraphs(data: bytes) -> list[str]:
    from lxml import html as lh

    root = lh.fromstring(data)
    for bad in root.xpath("//script|//style|//noscript|//head|//nav|//footer"):
        bad.getparent().remove(bad)
    out: list[str] = []

    def walk(el) -> None:
        tag = el.tag if isinstance(el.tag, str) else ""
        kids = [k for k in el if isinstance(k.tag, str)]
        has_block_child = any(k.tag in _BLOCK_TAGS or k.tag in ("ul", "ol", "table") for k in kids)
        if tag in _BLOCK_TAGS and not has_block_child:
            s = re.sub(r"\s+", " ", el.text_content()).strip()
            if s:
                out.append(s)
            return
        if el.text and el.text.strip() and has_block_child:
            out.append(re.sub(r"\s+", " ", el.text).strip())
        for k in kids:
            walk(k)
            if k.tail and k.tail.strip():
                out.append(re.sub(r"\s+", " ", k.tail).strip())

    walk(root)
    return out


def _text_paragraphs(text: str) -> list[str]:
    paras = []
    for chunk in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
        cur: list[str] = []
        for ln in lines:  # a list marker or heading line starts a new paragraph
            if cur and (_ITEM_RE.match(ln) or _heading(ln)):
                paras.append(" ".join(cur))
                cur = []
            cur.append(ln)
        if cur:
            paras.append(" ".join(cur))
    return [re.sub(r"^#{1,6}\s+", "", p) for p in paras]


def read_paragraphs(path: Path, language: str = "auto") -> tuple[list[str], str]:
    """Paragraphs of the file and the resolved language (``en`` / ``th``)."""
    suf = path.suffix.lower()
    if suf not in SUFFIXES:
        raise ValueError(f"unsupported file type {suf!r}; use one of {sorted(SUFFIXES)}")
    if suf == ".pdf":
        return _pdf_paragraphs(path.read_bytes(), language)
    if suf in (".html", ".htm"):
        paras = _html_paragraphs(path.read_bytes())
    else:
        paras = _text_paragraphs(path.read_text(encoding="utf-8", errors="replace"))
    return paras, detect_language("\n".join(paras), language)


# ---------------------------------------------------------------- paragraphs -> units


@dataclass
class _Level:
    rank: int
    token: str
    title: str


def _heading(p: str) -> tuple[int, str] | None:
    for rx, rank in _HEADINGS:
        m = rx.match(p)
        if m:
            kind, num = m.group(1), m.group(2).translate(_TH_DIGITS)
            return rank, re.sub(r"[\s/]", "", f"{kind}{num}").rstrip(".")
    return None


def split_long(p: str, limit: int = MAX_UNIT_CHARS) -> list[str]:
    """Split a long paragraph at sentence ends (else spaces, else hard cut) into pieces <= limit."""
    if len(p) <= limit:
        return [p]
    pieces: list[str] = []
    cur = ""
    for sent in _SENT_END.split(p):
        for part in [sent] if len(sent) <= limit else re.split(r"(?<=\s)", sent):
            while len(part) > limit:  # no spaces at all: hard cut
                pieces.append(part[:limit])
                part = part[limit:]
            if cur and len(cur) + 1 + len(part) > limit:
                pieces.append(cur.strip())
                cur = ""
            cur = f"{cur} {part}".strip() if cur else part
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def segment(paragraphs: list[str], language: str, version: str = "") -> list[Unit]:
    """Paragraphs -> units with hierarchical ids (see module docstring)."""
    stack: list[_Level] = []
    counters: dict[str, int] = {}
    lead_in = ""
    units: list[Unit] = []
    for raw in paragraphs:
        p = re.sub(r"\s+", " ", raw).strip()
        if not p or _PAGE_NO_RE.match(p):
            continue
        h = _heading(p)
        heading_only = False
        if h:
            rank, token = h
            stack = [lv for lv in stack if lv.rank < rank]
            stack.append(_Level(rank, token, p[:_MAX_HEADING_LEN]))
            lead_in = ""
            heading_only = len(p) <= _HEADING_ONLY_LEN and not p.endswith((".", ";", ":"))
            if heading_only:
                continue
        is_item = bool(_ITEM_RE.match(p))
        heading = lead_in if is_item and lead_in else (stack[-1].title if stack else "")
        key = "/".join(lv.token for lv in stack)
        for piece in split_long(p):
            counters[key] = counters.get(key, 0) + 1
            parts = [SOURCE, *([key] if key else []), str(counters[key])]
            units.append(
                Unit(
                    unit_id="/".join(parts),
                    source=SOURCE,
                    doc_id=SOURCE,
                    path=[lv.token for lv in stack],
                    heading=heading,
                    text=piece,
                    language=language,
                    version=version,
                )
            )
        if p.endswith(":"):
            lead_in = p
        elif not is_item:
            lead_in = ""
    return units


def segment_file(path: Path, language: str = "auto") -> list[Unit]:
    paras, lang = read_paragraphs(path, language)
    return segment(paras, lang)
