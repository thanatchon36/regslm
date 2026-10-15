"""EUR-Lex adapter: consolidated EU financial-services legislation (CRR, CRD, MiFID II, AMLD4).

Document types
    Consolidated versions ("CONVEX" documents, CELEX ``0YYYYTNNNN-YYYYMMDD``) of four base acts:
    CRR ``32013R0575``, CRD ``32013L0036``, MiFID II ``32014L0065``, AMLD4 ``32015L0849``.
    Consolidated texts are a documentation tool; only the OJ text is legally binding.

Access route (why Cellar and not the EUR-Lex website)
    ``eur-lex.europa.eu`` answers every automated request - including ``/robots.txt`` - with an
    AWS WAF JavaScript challenge (HTTP 202, empty body, ``x-amzn-waf-action: challenge``). We do not
    try to evade it (docs/sources/NEEDED.md). The Publications Office *Cellar* is the official
    machine route and is not challenged:

    1. Discovery: SPARQL GET ``https://publications.europa.eu/webapi/rdf/sparql`` lists the
       consolidated works of a base act (``cdm:act_consolidated_based_on_resource_legal``) with
       their ``cdm:act_consolidated_date``. The latest version dated <= today is used (versions
       with a future date, i.e. already adopted but not yet applicable, are ignored).
    2. Content: GET ``http://publications.europa.eu/resource/cellar/<uuid>`` with
       ``Accept: application/xhtml+xml`` and ``Accept-Language: <eng|fra|...>``; Cellar answers
       303 -> ``.../<uuid>.0006.03/DOC_1`` (the XHTML manifestation).

    Both calls go through a ``PoliteClient`` subclass that adds the negotiation headers while still
    using the disk cache, robots.txt check and the >= 2 s per-host rate limit. The cache key
    includes the headers, so each language is cached separately. robots.txt of
    ``publications.europa.eu`` redirects to ``op.europa.eu`` and does not disallow these paths.

Structure of the XHTML
    Part (``prt_``) > Title (``tis_``) > Chapter (``cpt_``) > Section (``sct_``) > Sub-section
    (``sbs_``) > Article (``div.eli-subdivision#art_N``) > paragraph (``div.norm`` with
    ``span.no-parag``) > point (``div.grid-container.grid-list``, label ``(a)``) > sub-point
    ``(i)``; annexes are ``div#anx_*``. Consolidated texts contain no preamble/recitals, so
    ``last_stats.recitals_skipped`` is normally 0 (counts recital-like blocks, should any appear).
    Amendments are marked with ``p.modref`` anchors (``▼M8`` etc.; ``M`` = amendment,
    ``C`` = corrigendum, ``A`` = accession, ``B`` = base text) that precede the amended block.

Units
    Smallest numbered item that carries text: a paragraph without points, or each point /
    sub-point (leaf). Lead-in sentences of non-leaf items are not duplicated into ``text``; they are
    carried in ``heading`` as ``"<article title> | <lead-in> | <lead-in> ..."``. Unnumbered text
    blocks get label ``pN`` when the article also has numbered paragraphs or several blocks.
    Annex items are numbered items; table rows become units ``T<n>R<m>`` (cells joined by `` | ``);
    correlation/transposition-table annexes are skipped. unit_id =
    ``eurlex/<doc>/Art92/1/a/i`` (``AnnexI/3/a``); ``<doc>`` gets a ``-<lang>`` suffix when the
    language is not ``en``.

``last_amended``
    Latest date among the ``M``/``A`` amendment markers found in the article (including the marker
    that precedes the article block), mapped to the amending act's date via the "Amended by"
    header table (English: the act date "of 26 June 2013"; otherwise the OJ publication date).
    Articles never amended carry the date of the base act. Unmappable -> consolidation date.

Languages
    ``languages=["en"]`` by default; any of the 24 official EU language codes (2-letter).

Licence
    Commission Decision 2011/833/EU on the reuse of Commission documents; legal notice
    https://eur-lex.europa.eu/content/legal-notice/legal-notice.html. Attribution required:
    "(c) European Union, https://eur-lex.europa.eu/". redistributable and derivative_ok.

Default selection: CRR, CRD, MiFID2, AMLD4 (``selection`` takes these doc_ids).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx
from lxml import html as lxml_html  # type: ignore[import-untyped]

from regslm.corpus.base import (
    Document,
    Licence,
    PoliteClient,
    RawDoc,
    SourceAdapter,
    Unit,
    make_unit_id,
    sha256_bytes,
)

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
CELLAR_ACCEPT = "application/xhtml+xml"

INSTRUMENTS: dict[str, tuple[str, str]] = {
    # doc_id -> (base CELEX, short title)
    "CRR": ("32013R0575", "Capital Requirements Regulation"),
    "CRD": ("32013L0036", "Capital Requirements Directive"),
    "MiFID2": ("32014L0065", "Markets in Financial Instruments Directive II"),
    "AMLD4": ("32015L0849", "Fourth Anti-Money Laundering Directive"),
}
_CELEX_TO_DOC = {v[0]: k for k, v in INSTRUMENTS.items()}

# 2-letter EU language code -> ISO 639-3 code used by Cellar content negotiation
LANG3: dict[str, str] = {
    "bg": "bul", "cs": "ces", "da": "dan", "de": "deu", "el": "ell", "en": "eng",
    "es": "spa", "et": "est", "fi": "fin", "fr": "fra", "ga": "gle", "hr": "hrv",
    "hu": "hun", "it": "ita", "lt": "lit", "lv": "lav", "mt": "mlt", "nl": "nld",
    "pl": "pol", "pt": "por", "ro": "ron", "sk": "slk", "sl": "slv", "sv": "swe",
}  # fmt: skip

LICENCE = Licence(
    name="EU Commission Decision 2011/833/EU (reuse of Commission documents)",
    url="https://eur-lex.europa.eu/content/legal-notice/legal-notice.html",
    redistributable=True,
    derivative_ok=True,
    attribution="© European Union, https://eur-lex.europa.eu/",
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"],
        start=1,
    )
}  # fmt: skip
_MARKER_RE = re.compile(r"[►▼◄]\s*([A-Z]\d*)")
_STRAY_MARK_RE = re.compile(r"[►▼◄]")
_EMPTY_RE = re.compile(
    r"^\[?\s*\(?(reserved|deleted|repealed|omitted|intentionally left blank|not used)\)?\s*\]?\.?$",
    re.I,
)
_INLINE_TAGS = {"a", "span", "sup", "sub", "i", "b", "em", "strong", "u", "img", "abbr", "small"}
_SKIP_CLASS_PREFIXES = ("title-", "hd-", "toc-", "stitle-", "footnote", "disclaimer", "reference")
_DIVISION_ID = re.compile(r"^(prt|tis|cpt|sct|sbs|dvs|ssb)_")
_NUMBERED_ITEM = re.compile(r"^(\d+[a-z]{0,2})\.\s+(.*)$", re.S)
_SKIP_ANNEX_TITLE = re.compile(r"correlation|concordance|transposition", re.I)


class EurLexBlockedError(RuntimeError):
    """Raised when EUR-Lex / Cellar cannot be reached without evading a block."""


class _CellarClient(PoliteClient):
    """PoliteClient that can send content-negotiation headers (cached per header set)."""

    def get_negotiated(self, url: str, headers: dict[str, str], suffix: str = "") -> RawDoc:
        tag = url + "|" + "|".join(f"{k.lower()}={v}" for k, v in sorted(headers.items()))
        key = self._key(tag)
        body = self.cache_dir / f"{key}{suffix}"
        meta = self.cache_dir / f"{key}.meta.json"
        if body.exists() and meta.exists():
            return RawDoc.model_validate_json(meta.read_text())
        if not self._allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        self._wait(url)
        r = self._http.get(url, headers=headers)
        if r.status_code in (202, 403, 429):
            raise EurLexBlockedError(
                f"{url} answered HTTP {r.status_code} (WAF/challenge or rate limit); "
                "see docs/sources/NEEDED.md"
            )
        r.raise_for_status()
        body.write_bytes(r.content)
        raw = RawDoc(
            source=self.source,
            url=url,
            sha256=sha256_bytes(r.content),
            path=str(body),
            content_type=r.headers.get("content-type", ""),
            fetched_at=datetime.now(UTC),
        )
        meta.write_text(raw.model_dump_json())
        return raw


@dataclass
class Parsed:
    """Extra parse statistics (kept on the adapter as ``last_stats``)."""

    recitals_skipped: int = 0
    annexes_skipped: list[str] = field(default_factory=list)
    articles: int = 0


@dataclass
class _Item:
    kind: str  # text | para | list | table
    label: str = ""
    text: str = ""
    node: lxml_html.HtmlElement | None = None


def _clean(s: str) -> str:
    s = s.replace("\xa0", " ").replace("​", "")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r" +([,;:.)])", r"\1", s)


def _parse_date_en(text: str) -> str | None:
    m = re.search(r"\bof (\d{1,2}) ([A-Za-z]+) (\d{4})", text)
    if m and m.group(2).lower() in _MONTHS:
        return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))).isoformat()
    return None


def _parse_date_dotted(text: str) -> str | None:
    ms = re.findall(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text)
    if not ms:
        return None
    d, mo, y = ms[-1]
    return date(int(y), int(mo), int(d)).isoformat()


def _label_token(label: str) -> str:
    if re.match(r"^-\d", label):
        return "m" + re.sub(r"\W+", "", label[1:])
    if label.startswith(("—", "–")) or label == "-":
        return "d" + re.sub(r"\W+", "", label)
    return re.sub(r"[^\w]+", "", label)


class _Parser:
    def __init__(
        self, doc_id: str, lang: str, version: str, amend_dates: dict[str, str], base_date: str
    ) -> None:
        self.doc_id = doc_id
        self.lang = lang
        self.version = version
        self.amend_dates = amend_dates
        self.base_date = base_date
        self.units: list[Unit] = []
        self.stats = Parsed()
        self._seen: set[str] = set()
        self._pending: set[str] = set()  # markers seen outside any article block
        self._markers: set[str] = set()  # markers collected for the current article/annex
        self._buffer: list[tuple[list[str], str, str]] = []
        self._in_annex = False

    # ----- text -------------------------------------------------------------------------
    def _inline(self, el: lxml_html.HtmlElement) -> str:
        parts: list[str] = [el.text or ""]
        for c in el:
            tag = c.tag if isinstance(c.tag, str) else ""
            if tag == "a":
                ctext = c.text_content()
                href = c.get("href") or ""
                m = _MARKER_RE.search(ctext)
                if m and len(_clean(ctext)) <= 6:
                    self._markers.add(m.group(1))
                    parts.append(c.tail or "")
                    continue
                if href.startswith("#E") or (c.get("id") or "").startswith("src."):
                    parts.append(c.tail or "")  # footnote marker
                    continue
            if tag in _INLINE_TAGS:
                parts.append(self._inline(c))
            else:
                parts.append(" " + self._inline(c) + " ")
            parts.append(c.tail or "")
        return "".join(parts)

    def _text(self, el: lxml_html.HtmlElement) -> str:
        s = self._inline(el)
        s = _STRAY_MARK_RE.sub(" ", s)
        s = re.sub(r"\(\s*\)", "", s)
        return _clean(s)

    # ----- item scanning ----------------------------------------------------------------
    def _is_inline(self, c: lxml_html.HtmlElement) -> bool:
        return isinstance(c.tag, str) and c.tag in _INLINE_TAGS

    def _items(self, el: lxml_html.HtmlElement) -> list[_Item]:
        out: list[_Item] = []
        buf: list[str] = []

        def flush() -> None:
            raw = "".join(buf)
            buf.clear()
            t = _clean(_STRAY_MARK_RE.sub(" ", re.sub(r"\(\s*\)", "", raw)))
            if t:
                out.append(_Item("text", text=t))

        if el.text:
            buf.append(el.text)
        for c in el:
            if self._is_inline(c):
                buf.append(self._inline(c))
            elif isinstance(c.tag, str):
                flush()
                out.extend(self._block(c))
            if c.tail:
                buf.append(c.tail)
        flush()
        return out

    def _block(self, c: lxml_html.HtmlElement) -> list[_Item]:
        tag = c.tag
        cls = c.get("class") or ""
        cid = c.get("id") or ""
        if tag in ("hr", "br", "col", "script", "style"):
            return []
        if tag == "p" and "modref" in cls.split():
            self._markers.update(m for m in _MARKER_RE.findall(c.text_content()))
            return []
        if tag == "p" and self._in_annex and "title-gr-seq-level-2" in cls.split():
            t = self._text(c)
            return [_Item("heading", text=t)] if t else []
        if cls.startswith(_SKIP_CLASS_PREFIXES) or cid.endswith(".tit_1"):
            return []
        if tag == "table":
            return [_Item("table", node=c)]
        if tag == "div" and "grid-list" in cls.split() and "grid-container" in cls.split():
            kids = [k for k in c if isinstance(k.tag, str)]
            if len(kids) >= 2:
                return [_Item("list", label=_clean(kids[0].text_content()), node=kids[1])]
        if tag == "div" and "norm" in cls.split():
            first = next((k for k in c if isinstance(k.tag, str)), None)
            if first is not None and "no-parag" in (first.get("class") or ""):
                inner = next((k for k in c if k is not first and isinstance(k.tag, str)), None)
                return [_Item("para", label=_clean(first.text_content()), node=inner)]
        if tag == "p":
            t = self._text(c)
            return [_Item("text", text=t)] if t else []
        if tag == "div":
            return self._items(c)
        t = self._text(c)
        return [_Item("text", text=t)] if t else []

    # ----- emission ---------------------------------------------------------------------
    def _emit(self, path: list[str], leadins: list[str], text: str) -> None:
        text = _clean(text)
        if not re.search(r"\w", text) or _EMPTY_RE.match(text):
            return
        self._buffer.append((list(path), " | ".join(leadins), text))

    def _process(self, items: list[_Item], base0: list[str], lead0: list[str]) -> None:
        base, leadins = base0, lead0
        if self._in_annex:  # annex items are plain "1. text" blocks
            fixed: list[_Item] = []
            for it in items:
                m = _NUMBERED_ITEM.match(it.text) if it.kind == "text" else None
                fixed.append(_Item("para", label=m.group(1) + ".", text=m.group(2)) if m else it)
            items = fixed
        dashes = 0
        for it in items:  # bullets ("—") get a sibling index so that unit ids stay unique
            if it.kind in ("list", "para") and it.label in ("—", "–", "-", "•"):
                dashes += 1
                it.label = f"— {dashes}"
        has_numbered = any(i.kind == "para" for i in items)
        k = 0
        i = 0
        while i < len(items):
            it = items[i]
            if it.kind == "heading":  # annex "SECTION A" / sub-title
                if re.match(r"^(SECTION|PART|CHAPTER|TITLE)\b", it.text, re.I):
                    base, leadins = [*base0, it.text], lead0
                else:
                    leadins = [*lead0, it.text]
                i += 1
                continue
            if it.kind == "text" or (it.kind == "para" and it.node is None):
                if it.kind == "text":
                    k += 1
                    sub = "" if (k == 1 and not has_numbered) else f"p{k}"
                else:
                    sub = it.label
                path = base + ([sub] if sub else [])
                j = i + 1
                kids: list[_Item] = []
                while j < len(items) and items[j].kind in ("list", "table"):
                    kids.append(items[j])
                    j += 1
                if kids:
                    for kid in kids:
                        assert kid.node is not None
                        if kid.kind == "table":
                            self._table(kid.node, path, [*leadins, it.text])
                            continue
                        self._process(
                            self._items(kid.node), [*path, kid.label], [*leadins, it.text]
                        )
                else:
                    self._emit(path, leadins, it.text)
                i = j
                continue
            if it.kind in ("para", "list") and it.node is not None:
                self._process(self._items(it.node), [*base, it.label], leadins)
            elif it.kind == "table" and it.node is not None:
                self._table(it.node, base, leadins)
            i += 1

    def _table(self, table: lxml_html.HtmlElement, base: list[str], leadins: list[str]) -> None:
        key = tuple(base)
        n = self._table_count.get(key, 0) + 1
        self._table_count[key] = n
        for m, tr in enumerate(table.iter("tr"), start=1):
            tds = [td for td in tr if td.tag in ("td", "th")]
            if tds and all("font-weight: bold" in (td.get("style") or "") for td in tds):
                continue
            cells = [self._text(td) for td in tds]
            row = " | ".join(c for c in cells if c)
            if len(row) >= 3:
                self._emit([*base, f"T{n}R{m}"], leadins, row)

    # ----- flush a finished article / annex ---------------------------------------------
    def _flush(self, id_root: str, last_amended: str) -> None:
        for path, heading, text in self._buffer:
            tokens = [_label_token(p) for p in path]
            uid = make_unit_id("eurlex", self.doc_id, id_root, *tokens[self._depth + 1 :])
            base_uid, c = uid, 2
            while uid in self._seen:
                uid, c = f"{base_uid}~{c}", c + 1
            self._seen.add(uid)
            self.units.append(
                Unit(
                    unit_id=uid,
                    source="eurlex",
                    doc_id=self.doc_id,
                    path=path,
                    heading=heading,
                    text=text,
                    language=self.lang,
                    version=self.version,
                    last_amended=last_amended,
                )
            )
        self._buffer = []

    def _amended(self) -> str:
        dates = [self.amend_dates[m] for m in self._markers if m in self.amend_dates]
        return max(dates) if dates else self.base_date

    # ----- structure --------------------------------------------------------------------
    def run(self, body: lxml_html.HtmlElement) -> None:
        self._table_count: dict[tuple[str, ...], int] = {}
        self._depth = 0
        self._walk(body, [])

    def _walk(self, el: lxml_html.HtmlElement, div_path: list[str]) -> None:
        for c in el:
            if not isinstance(c.tag, str):
                continue
            cls = c.get("class") or ""
            cid = c.get("id") or ""
            if c.tag == "p" and "modref" in cls.split():
                self._pending.update(_MARKER_RE.findall(c.text_content()))
            elif c.tag == "div" and cid.startswith("art_") and "." not in cid:
                self._article(c, div_path)
            elif c.tag == "div" and cid.startswith("anx_"):
                self._annex(c)
            elif c.tag == "div" and _DIVISION_ID.match(cid):
                lab = c.find(".//p[@class='title-division-1']")
                label = _clean(lab.text_content()) if lab is not None else cid
                self._walk(c, [*div_path, self._strip_markers(label)])
            elif c.tag == "div" and (
                "eli-container" in cls or cid.startswith("enc_") or cid.startswith("cit_")
            ):
                self._walk(c, div_path)
            elif cid.startswith(("rct_", "pbl_")) or "recital" in cls:
                self.stats.recitals_skipped += 1

    @staticmethod
    def _strip_markers(s: str) -> str:
        return _clean(_STRAY_MARK_RE.sub(" ", _MARKER_RE.sub(" ", s)))

    def _article(self, art: lxml_html.HtmlElement, div_path: list[str]) -> None:
        self._markers = set(self._pending)
        self._pending = set()
        head = art.find("p[@class='title-article-norm']")
        label = self._strip_markers(_clean(head.text_content())) if head is not None else ""
        m = re.search(r"(\d+[A-Za-z]*)\s*$", label)
        num = m.group(1) if m else _label_token(label)
        t = art.find(".//p[@class='stitle-article-norm']")
        title = self._text(t) if t is not None else ""
        if t is not None:
            self._markers.update(_MARKER_RE.findall(t.text_content()))
        path = [*div_path, f"Article {num}"]
        self._depth = len(div_path)
        self._process(self._items(art), path, [title] if title else [])
        self.stats.articles += 1
        self._flush(f"Art{num}", self._amended())
        self._markers = set()

    def _annex(self, anx: lxml_html.HtmlElement) -> None:
        self._markers = set(self._pending)
        self._pending = set()
        t1 = anx.find(".//p[@class='title-annex-1']")
        label = self._strip_markers(_clean(t1.text_content())) if t1 is not None else "ANNEX"
        t2 = anx.xpath(".//p[@class='title-annex-1']/following-sibling::p[1]")
        title = self._text(t2[0]) if t2 else ""
        if _SKIP_ANNEX_TITLE.search(title):
            self.stats.annexes_skipped.append(label)
            self._markers = set()
            return
        token = "Annex" + _label_token(label.split()[-1]) if " " in label else "Annex"
        self._in_annex = True
        self._depth = 0
        items = self._items(anx)
        if title:  # the annex title paragraph is already carried in the heading
            items = [i for i in items if not (i.kind == "text" and i.text == title)]
        self._process(items, [label], [title] if title else [])
        self._in_annex = False
        self._flush(token, self._amended())
        self._markers = set()


def _amendment_dates(tree: lxml_html.HtmlElement) -> tuple[dict[str, str], str]:
    """Map ``M1``.. -> date from the 'Amended by' header table; also the base act date."""
    dates: dict[str, str] = {}
    for tr in tree.iter("tr"):
        cells = [_clean(td.text_content()) for td in tr if td.tag in ("td", "th")]
        if not cells:
            continue
        m = re.match(r"^[►▼◄\s]*([MA]\d+)\b", " ".join(cells[:2]))
        if not m:
            continue
        row = " ".join(cells)
        d = _parse_date_en(row) or _parse_date_dotted(row)
        if d:
            dates[m.group(1)] = d
    base = ""
    for p in tree.iterfind(".//p[@class='title-doc-first']"):
        d = _parse_date_en(" " + _clean(p.text_content()))
        if d:
            base = d
            break
    return dates, base


class EurLexAdapter(SourceAdapter):
    name = "eurlex"
    licence = LICENCE
    client: _CellarClient

    def __init__(
        self,
        raw_root: Path = Path("data/raw"),
        languages: list[str] | None = None,
        as_of: date | None = None,
    ) -> None:
        super().__init__(raw_root)
        self.client = _CellarClient(cache_dir=raw_root / self.name, source=self.name)
        self.languages = languages or ["en"]
        for lg in self.languages:
            if lg not in LANG3:
                raise ValueError(f"unsupported EU language code {lg!r}")
        self.as_of = as_of or date.today()
        self.last_stats: Parsed = Parsed()

    # ----- discovery ----------------------------------------------------------------------
    def sparql_url(self, celex: str) -> str:
        q = (
            "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#> "
            "SELECT ?c ?d WHERE { "
            f'?w cdm:resource_legal_id_celex "{celex}"^^<http://www.w3.org/2001/XMLSchema#string> .'
            " "
            "?c cdm:act_consolidated_based_on_resource_legal ?w . "
            "?c cdm:act_consolidated_date ?d } ORDER BY DESC(?d)"
        )
        return SPARQL_ENDPOINT + "?" + urlencode({"query": q})

    def consolidated_versions(self, celex: str) -> list[tuple[str, str]]:
        """(cellar work URI, ISO date) of consolidations dated <= as_of, newest first."""
        raw = self.client.get_negotiated(
            self.sparql_url(celex), {"Accept": "application/sparql-results+json"}, ".json"
        )
        data = json.loads(self.client.read(raw))
        found = {
            (b["c"]["value"], b["d"]["value"])
            for b in data["results"]["bindings"]
            if b["d"]["value"] <= self.as_of.isoformat()
        }
        if not found:
            raise RuntimeError(f"no consolidated version of {celex} found via Cellar SPARQL")
        return sorted(found, key=lambda t: t[1], reverse=True)

    # ----- SourceAdapter ------------------------------------------------------------------
    def fetch(self, selection: list[str] | None = None) -> Iterator[RawDoc]:
        for doc_id in selection or list(INSTRUMENTS):
            if doc_id not in INSTRUMENTS:
                raise KeyError(
                    f"unknown EUR-Lex instrument {doc_id!r}; choose from {list(INSTRUMENTS)}"
                )
            versions = self.consolidated_versions(INSTRUMENTS[doc_id][0])
            for lg in self.languages:
                headers = {"Accept": CELLAR_ACCEPT, "Accept-Language": LANG3[lg]}
                for uri, _date in versions:  # newest consolidation that has this language
                    try:
                        yield self.client.get_negotiated(uri, headers, ".xhtml")
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            continue
                        raise
                    break
                else:
                    raise RuntimeError(f"no {lg} consolidated text for {doc_id} in Cellar")

    def parse(self, raw: RawDoc) -> tuple[Document, list[Unit]]:
        data = self.client.read(raw)
        tree = lxml_html.fromstring(data)
        title_el = tree.find(".//title")
        title_txt = _clean(title_el.text_content()) if title_el is not None else ""
        m = re.search(
            r"TEXT:\s*(\d{5}[A-Z]\d{4})\s*—\s*([A-Z]{2})\s*—\s*(\d{2})\.(\d{2})\.(\d{4})", title_txt
        )
        if not m:
            raise ValueError(f"not a Cellar consolidated text (title: {title_txt!r})")
        base_celex, lang = m.group(1), m.group(2).lower()
        base_celex = "3" + base_celex[1:]
        version = f"{m.group(5)}-{m.group(4)}-{m.group(3)}"
        key = _CELEX_TO_DOC.get(base_celex)
        if key is None:
            raise ValueError(f"unknown instrument {base_celex}")
        doc_id = key if lang == "en" else f"{key}-{lang}"
        amend, base_date = _amendment_dates(tree)
        parser = _Parser(doc_id, lang, version, amend, base_date or version)
        body = tree.find("body")
        parser.run(body)
        self.last_stats = parser.stats
        mt = tree.find(".//div[@class='eli-main-title']")
        title = (
            self._strip(mt.text_content()) if mt is not None else INSTRUMENTS[key][1]
        ) or INSTRUMENTS[key][1]
        doc = Document(
            source=self.name,
            doc_id=doc_id,
            title=title,
            language=lang,
            url=(
                "https://eur-lex.europa.eu/legal-content/"
                f"{lang.upper()}/TXT/?uri=CELEX:0{base_celex[1:]}-{version.replace('-', '')}"
            ),
            version=version,
            sha256=raw.sha256,
        )
        return doc, parser.units

    @staticmethod
    def _strip(s: str) -> str:
        return _Parser._strip_markers(s)
