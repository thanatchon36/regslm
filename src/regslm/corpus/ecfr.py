"""eCFR adapter: US Code of Federal Regulations (same XML as ecfr.gov) via GovInfo bulk data.

Route choice (robots.txt)
    ``www.ecfr.gov/robots.txt`` disallows ``/api/versioner/v1/full/`` (the per-part
    full-text API named in the first spec draft), so that endpoint is NOT used.
    Text comes from the GovInfo bulk files instead (govinfo.gov robots.txt advertises
    ``/bulkdata/ECFR/`` sitemaps): one whole-title XML per title, fetched once and
    cached; the selected parts are cut out at parse time. Amendment dates for
    ``Unit.last_amended`` come from the (robots-permitted) eCFR
    ``/api/versioner/v1/versions/title-{t}.json?part={p}`` endpoint; if that is
    unreachable, ``last_amended`` is None.

Document types
    One ``Document`` per selected CFR part (``doc_id`` = ``{title}CFR{part}``, e.g.
    ``12CFR217``); for section selections (17 CFR 240) one Document holding only the
    chosen sections. ``fetch`` yields one RawDoc per *title*, so a RawDoc maps to
    several Documents: use ``parse_all(raw)``; ``parse(raw)`` works only when a single
    document is selected from that title.

URL patterns
    ``https://www.govinfo.gov/bulkdata/ECFR/title-{t}/ECFR-title{t}.xml``
    ``https://www.ecfr.gov/api/versioner/v1/versions/title-{t}.json?part={p}`` (or ``section=``)

Version
    ``<AMDDATE>`` in the bulk file header (the date the file is current through),
    e.g. 2026-08-17. NB: the bulk files can lag ecfr.gov by weeks.

Structure
    part (DIV5) -> subpart (DIV6) -> [subject group (DIV7)] -> section (DIV8) /
    appendix (DIV9) -> flat ``<P>`` elements. eCFR flattens paragraphs, so nesting
    is rebuilt from the leading markers: ``(a)`` -> ``(1)`` -> ``(i)`` -> ``(A)`` ->
    italic ``(1)`` -> italic ``(i)``. A marker such as ``(i)`` that could be a letter
    or a roman numeral is resolved from the markers already open (is it the successor
    of the last letter / of the last roman numeral?) and, when still ambiguous,
    from whether the previous paragraph ends with ``:`` (a lead-in => child).

Units
    One unit per ``<P>`` (paragraph). Leading markers are stripped from ``text``
    and live in ``path`` / ``unit_id``. ``heading`` = section heading plus the
    lead-ins / run-in headings of the ancestor paragraphs. Heading-only paragraphs,
    "[Reserved]" / "Deleted" paragraphs, tables, footnote bodies, source notes
    (``CITA``) and images are not emitted. Unnumbered paragraphs (continuation
    text, ``<EXTRACT>`` content, ``FP*`` flush paragraphs) are emitted as ``p1``,
    ``p2``... under the currently open paragraph without altering the nesting.

Language: English (``en``).

Licence: works of the US federal government are in the public domain
(17 U.S.C. 105, https://www.law.cornell.edu/uscode/text/17/105);
https://www.ecfr.gov/reader-aids/understanding-the-ecfr/ecfr-copyright says the
same. Redistribution and derivative use are unrestricted.

Default selection (``DEFAULT_SELECTION``, overridable via ``selection``):
    Title 12 parts 3, 217, 249, 252, 324, 1005, 1026; Title 17 part 275 and the
    sections 240.15c3-1, 240.15c3-3, 240.17a-3, 240.17a-4 of part 240.
    Selection strings: ``"12CFR217"`` (whole part) or ``"17CFR240:240.15c3-1"``
    (one section).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import httpx
from lxml import etree  # type: ignore[import-untyped]

from regslm.corpus.base import (
    Document,
    Licence,
    RawDoc,
    SourceAdapter,
    Unit,
    make_unit_id,
)

BULK = "https://www.govinfo.gov/bulkdata/ECFR/title-{t}/ECFR-title{t}.xml"
VERSIONS = "https://www.ecfr.gov/api/versioner/v1/versions/"

DEFAULT_SELECTION: list[str] = [
    "12CFR3",
    "12CFR217",
    "12CFR249",
    "12CFR252",
    "12CFR324",
    "12CFR1005",
    "12CFR1026",
    "17CFR240:240.15c3-1",
    "17CFR240:240.15c3-3",
    "17CFR240:240.17a-3",
    "17CFR240:240.17a-4",
    "17CFR275",
]

_SEL_RE = re.compile(r"^(\d+)CFR(\d+)(?::(.+))?$")
_MARKER_RE = re.compile(r"\s*\(([A-Za-z]{1,6}|\d{1,3})\)")
_SKIP_TEXT_RE = re.compile(
    r"^[\[\(]?\s*(reserved|removed|deleted|removed and reserved|omitted)\s*[\]\)]?\.?$", re.I
)
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_SKIP_TAGS = {"SU", "FTREF", "img", "CITA", "FTNT"}

# nesting levels
L_ALPHA, L_DIGIT, L_ROMAN, L_UPPER, L_IDIGIT, L_IROMAN = range(6)


def _roman_val(s: str) -> int | None:
    s = s.lower()
    if not s or any(c not in _ROMAN for c in s):
        return None
    total = 0
    for i, c in enumerate(s):
        v = _ROMAN[c]
        total += -v if i + 1 < len(s) and _ROMAN[s[i + 1]] > v else v
    return total


def _to_roman(n: int) -> str:
    out = ""
    for v, s in ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out += s
            n -= v
    return out


def _alpha_next(s: str) -> str:
    """a->b, z->aa, aa->bb, zz->aaa (CFR double-letter convention)."""
    if not s:
        return "a"
    c = s[0]
    if c == "z":
        return "a" * (len(s) + 1)
    return chr(ord(c) + 1) * len(s)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _segments(el: etree._Element) -> list[tuple[str, bool]]:
    """Flatten an element to (text, italic) runs; footnote refs/images dropped."""
    out: list[tuple[str, bool]] = []

    def walk(e: etree._Element, ital: bool) -> None:
        if e.text:
            out.append((e.text, ital))
        for ch in e:
            tag = ch.tag if isinstance(ch.tag, str) else ""
            if tag == "br":
                out.append((" ", ital))
            elif tag not in _SKIP_TAGS:
                walk(ch, ital or tag == "I" or (tag == "E" and ch.get("T") == "03"))
            if ch.tail:
                out.append((ch.tail, ital))

    walk(el, False)
    return out


def _flat(segs: list[tuple[str, bool]]) -> tuple[str, list[bool]]:
    chars: list[str] = []
    flags: list[bool] = []
    for t, it in segs:
        t = t.replace("\n", " ").replace("\xa0", " ")
        chars.extend(t)
        flags.extend([it] * len(t))
    return "".join(chars), flags


class _Para:
    """A parsed paragraph: markers (label, level) + text + heading-only flag."""

    def __init__(self, el: etree._Element) -> None:
        raw, flags = _flat(_segments(el))
        pos = 0
        # (token, italic, run-in heading that sits between this marker and the next)
        self.markers: list[tuple[str, bool, str]] = []
        while True:
            m = _MARKER_RE.match(raw, pos)
            if not m:
                break
            ital = all(flags[m.start(1) : m.end(1)])
            self.markers.append((m.group(1), ital, ""))
            pos = m.end(0)
            # "(a) <I>Heading.</I> (1) text": the italic run-in heading belongs to (a)
            j = pos
            while j < len(raw) and raw[j].isspace():
                j += 1
            h0 = j
            while j < len(raw) and flags[j]:
                j += 1
            if j > h0:
                k = j
                while k < len(raw) and (raw[k].isspace() or raw[k] in "\u2014\u2013-"):
                    k += 1
                if _MARKER_RE.match(raw, k):
                    tok, it, _ = self.markers[-1]
                    self.markers[-1] = (tok, it, _norm(raw[h0:j]))
                    pos = k
        rest_raw = raw[pos:]
        rest_flags = flags[pos:]
        self.text = _norm(rest_raw)
        visible = [f for c, f in zip(rest_raw, rest_flags, strict=True) if not c.isspace()]
        self.heading_only = bool(visible) and all(visible)
        self.full = _norm(raw)


class _Nester:
    """Reconstruct paragraph nesting from marker sequences within one section.

    Stack entries are ``(level, token, display)``. An unnumbered ``<P>`` (a
    definition, or a lead-in before ``(a)``) becomes an *anchor* at level -1
    that following first-of-sequence markers nest under.
    """

    FIRST: ClassVar[set[str]] = {"a", "1", "i", "A"}

    def __init__(self) -> None:
        self.stack: list[tuple[int, str, str]] = []
        self.saved: list[tuple[int, str, str]] | None = None
        self.prev_leadin = False  # previous paragraph ended with ':'

    def _last(self, level: int) -> str | None:
        for lv, tok, _ in self.stack:
            if lv == level:
                return tok
        return None

    def level_of(self, tok: str, italic: bool) -> int:
        if tok.isdigit():
            return L_IDIGIT if italic else L_DIGIT
        if tok.isupper():
            return L_UPPER
        rv = _roman_val(tok)
        if italic:
            return L_IROMAN
        if rv is None:
            return L_ALPHA
        last_a = self._last(L_ALPHA)
        exp_alpha = _alpha_next(last_a) if last_a else "a"
        last_r = self._last(L_ROMAN)
        exp_roman = _to_roman((_roman_val(last_r) or 0) + 1) if last_r else "i"
        is_a, is_r = tok == exp_alpha, tok == exp_roman
        if is_a and is_r:  # ambiguous, e.g. (h) ... (i): lead-in colon => child
            return L_ROMAN if self.prev_leadin else L_ALPHA
        if is_a:
            return L_ALPHA
        if is_r:
            return L_ROMAN
        if len(tok) > 1 and len(set(tok)) > 1:
            return L_ROMAN
        if tok == "i":
            return L_ROMAN if self.stack else L_ALPHA
        return L_ALPHA

    def anchor(self, display: str) -> None:
        if self.stack and self.stack[0][0] >= 0:
            self.saved = self.stack
        elif self.stack:  # previous anchor: keep what it displaced
            pass
        self.stack = [(-1, display, display)]

    def push(self, tok: str, italic: bool) -> None:
        if self.saved is not None and len(self.stack) <= 1:
            # right after an anchor: first-of-sequence markers nest under it,
            # anything else continues the sequence that was open before it
            if tok not in self.FIRST:
                self.stack, self.saved = self.saved, None
        elif self.saved is not None and len(self.stack) > 1:
            pass
        lv = self.level_of(tok, italic)
        while self.stack and self.stack[-1][0] >= lv:
            self.stack.pop()
        self.stack.append((lv, tok, f"({tok})"))


def _clean_heading(s: str) -> str:
    return _norm(s)


class EcfrAdapter(SourceAdapter):
    name = "ecfr"
    _selection: dict[int, dict[str, set[str] | None]]
    licence = Licence(
        name="Public domain (17 U.S.C. 105, US federal government work)",
        url="https://www.law.cornell.edu/uscode/text/17/105",
        redistributable=True,
        derivative_ok=True,
        attribution="U.S. Government Publishing Office / National Archives, eCFR (ecfr.gov)",
    )

    # ------------------------------------------------------------------ fetch
    @staticmethod
    def _parse_selection(selection: list[str] | None) -> dict[int, dict[str, set[str] | None]]:
        """title -> part -> None (whole part) | set of section numbers."""
        out: dict[int, dict[str, set[str] | None]] = {}
        for item in selection or DEFAULT_SELECTION:
            m = _SEL_RE.match(item)
            if not m:
                raise ValueError(
                    f"bad eCFR selection {item!r}; use '12CFR217' or '17CFR240:240.15c3-1'"
                )
            t, p, sec = int(m.group(1)), m.group(2), m.group(3)
            parts = out.setdefault(t, {})
            if sec is None:
                parts[p] = None
            elif parts.get(p, set()) is not None:
                cur = parts.setdefault(p, set())
                assert cur is not None
                cur.add(sec)
        return out

    def urls(self, selection: list[str] | None = None) -> list[str]:
        return [BULK.format(t=t) for t in self._parse_selection(selection)]

    def fetch(self, selection: list[str] | None = None) -> Iterator[RawDoc]:
        """One whole-title bulk XML per title (cached); parts are selected at parse time."""
        self._selection = self._parse_selection(selection)
        for t in self._selection:
            url = BULK.format(t=t)
            try:
                yield self.client.get(url, suffix=".xml")
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (403, 429):
                    raise RuntimeError(
                        f"{url} blocked (HTTP {e.response.status_code}); see docs/sources/NEEDED.md"
                    ) from e
                raise

    # ------------------------------------------------------------------ parse
    def _amendments(self, title: int, query: str) -> tuple[dict[str, list[str]], dict[str, str]]:
        """identifier -> sorted amendment dates (non-removed); identifier -> subpart."""
        url = f"{VERSIONS}title-{title}.json?{query}"
        dates: dict[str, list[str]] = {}
        subparts: dict[str, str] = {}
        try:
            raw = self.client.get(url, suffix=".json")
            rows = json.loads(self.client.read(raw)).get("content_versions", [])
        except (httpx.HTTPError, json.JSONDecodeError, PermissionError):
            return dates, subparts
        for r in rows:
            ident = r["identifier"]
            if r.get("subpart"):
                subparts[ident] = r["subpart"]
            if not r.get("removed"):
                dates.setdefault(ident, []).append(r["amendment_date"])
        for v in dates.values():
            v.sort()
        return dates, subparts

    @staticmethod
    def _amddate(path: Path) -> str:
        with path.open("rb") as f:
            head = f.read(8192).decode("utf-8", "replace")
        m = re.search(r"<AMDDATE>\s*([^<]+?)\s*</AMDDATE>", head)
        if not m:
            raise ValueError("no <AMDDATE> in bulk eCFR header")
        return (
            datetime.strptime(m.group(1).replace("Sept.", "Sep."), "%b. %d, %Y").date().isoformat()
        )

    def parse(self, raw: RawDoc) -> tuple[Document, list[Unit]]:
        """Parse a raw title when exactly one document is selected from it (else use parse_all)."""
        docs = self.parse_all(raw)
        if len(docs) != 1:
            raise ValueError(f"{raw.url} yields {len(docs)} documents; use parse_all()")
        return docs[0]

    def parse_all(
        self, raw: RawDoc, selection: list[str] | None = None
    ) -> list[tuple[Document, list[Unit]]]:
        """Parse one whole-title bulk file into one (Document, units) per selected part."""
        m = re.search(r"title-(\d+)/ECFR-title\d+\.xml", raw.url)
        if not m:
            raise ValueError(f"not an eCFR bulk URL: {raw.url}")
        title = int(m.group(1))
        sel = (
            self._parse_selection(selection)
            if selection is not None
            else getattr(self, "_selection", None) or self._parse_selection(None)
        ).get(title, {})
        version = self._amddate(Path(raw.path))
        results: list[tuple[Document, list[Unit]]] = []
        ctx = etree.iterparse(raw.path, events=("end",), tag="DIV5", huge_tree=True)
        for _, el in ctx:
            part = el.get("N") or ""
            if part in sel:
                results.append(self._parse_part(raw, title, part, el, sel[part], version))
            el.clear()
            while el.getprevious() is not None and el.getparent() is not None:
                del el.getparent()[0]
        found = {d.doc_id for d, _ in results}
        for part in sel:
            if f"{title}CFR{part}" not in found:
                raise ValueError(f"part {part} not found in title {title} bulk file")
        return results

    def _parse_part(
        self,
        raw: RawDoc,
        title: int,
        part: str,
        root: etree._Element,
        sections: set[str] | None,
        version: str,
    ) -> tuple[Document, list[Unit]]:
        doc_id = f"{title}CFR{part}"
        if sections is None:
            dates, subparts = self._amendments(title, f"part={part}")
        else:
            dates, subparts = {}, {}
            for sec in sorted(sections):
                d, sp = self._amendments(title, f"section={sec}")
                dates.update(d)
                subparts.update(sp)
        part_dates = sorted({d for v in dates.values() for d in v if d <= version})
        doc_level = part_dates[-1] if part_dates else None

        head = root.find("HEAD")
        doc_title = _norm("".join(head.itertext())) if head is not None else f"{title} CFR {part}"
        if sections is not None:
            doc_title += (
                " [selected sections: " + ", ".join(f"§ {s}" for s in sorted(sections)) + "]"
            )

        units: list[Unit] = []
        seen: set[str] = set()
        subpart_label: str | None = None
        for el in root.iter("DIV6", "DIV8", "DIV9"):
            if el.tag == "DIV6":
                subpart_label = f"Subpart {el.get('N')}"
                continue
            sec_n = (el.get("N") or "").replace("\xa7", "").strip()
            if sections is not None and sec_n not in sections:
                continue
            units.extend(
                self._section_units(
                    el,
                    sec_n=sec_n,
                    doc_id=doc_id,
                    part=part,
                    version=version,
                    subpart=subpart_label
                    or (f"Subpart {subparts[sec_n]}" if sec_n in subparts else None),
                    last=self._last_amended(dates.get(sec_n), version) or doc_level,
                    seen=seen,
                )
            )

        doc = Document(
            source=self.name,
            doc_id=doc_id,
            title=doc_title,
            language="en",
            url=raw.url,
            version=version,
            sha256=raw.sha256,
        )
        return doc, units

    @staticmethod
    def _last_amended(ds: list[str] | None, version: str) -> str | None:
        ok = [d for d in ds or [] if d <= version]
        return ok[-1] if ok else None

    def _section_units(
        self,
        el: etree._Element,
        *,
        doc_id: str,
        part: str,
        version: str,
        subpart: str | None,
        last: str | None,
        sec_n: str,
        seen: set[str],
    ) -> list[Unit]:
        head_el = el.find("HEAD")
        head = _clean_heading("".join(_flat(_segments(head_el))[0])) if head_el is not None else ""
        sec_label = _norm(sec_n or head.split("—")[0]) if el.tag == "DIV9" else f"§ {sec_n}"
        base_path = [f"Part {part}"] + ([subpart] if subpart else []) + [sec_label]

        # collect body paragraphs in document order: (element, numbered?)
        body: list[tuple[etree._Element, bool]] = []
        for ch in el:
            tag = ch.tag if isinstance(ch.tag, str) else ""
            if tag == "P":
                body.append((ch, True))
            elif tag in ("FP", "FP-1", "FP-2"):
                body.append((ch, False))
            elif tag == "EXTRACT":
                body.extend(
                    (p, False)
                    for p in ch
                    if isinstance(p.tag, str) and p.tag in ("P", "FP", "FP-1", "FP-2")
                )

        nester = _Nester()
        paras: list[tuple[_Para, str]] = []
        for p, num in body:
            para = _Para(p)
            kind = "marked" if num and para.markers else "anchor" if num else "side"
            paras.append((para, kind))
        n_anchor = sum(1 for _, k in paras if k == "anchor")
        out: list[Unit] = []
        counter = 0
        lead_by_depth: dict[int, tuple[str, str]] = {}  # stack index -> (label, lead-in)
        for para, kind in paras:
            if kind == "marked":
                inner: dict[int, str] = {}
                for tok, ital, h in para.markers:
                    nester.push(tok, ital)
                    if h:
                        inner[len(nester.stack) - 1] = h.rstrip(".:;\u2014\u2013 ")
                kept = nester.stack
                idx = len(kept) - 1
                for k in [k for k in lead_by_depth if k >= idx and k >= 0]:
                    del lead_by_depth[k]
                lead: str | None = None
                if para.heading_only:
                    lead = para.text.rstrip(".:; ")
                elif para.text.endswith(":"):
                    lead = para.text
                for k, h in inner.items():
                    if k < idx:
                        lead_by_depth[k] = (kept[k][2], h)
                if lead:
                    lead_by_depth[idx] = (kept[idx][2], lead)
                nester.prev_leadin = bool(lead)
                if para.heading_only or not para.text or _SKIP_TEXT_RE.match(para.text):
                    continue
                ancestors = [lead_by_depth[k] for k in sorted(lead_by_depth) if k < idx]
                levels = [d for _, _, d in kept if d]
                text = para.text
            else:
                if not para.full or _SKIP_TEXT_RE.match(para.full):
                    continue
                counter += 1
                text = para.full
                label = f"p{counter}"
                if kind == "anchor":
                    nester.anchor(label if n_anchor > 1 else "")
                    lead_by_depth.clear()
                    kept = nester.stack
                    idx = 0
                    ancestors = []
                    levels = [d for _, _, d in kept if d]
                    if text.endswith(":"):
                        lead_by_depth[0] = (label if n_anchor > 1 else "", text)
                else:  # side paragraph: child of the open paragraph, nesting untouched
                    kept = nester.stack
                    idx = len(kept)
                    ancestors = [lead_by_depth[k] for k in sorted(lead_by_depth) if k < idx]
                    levels = [d for _, _, d in kept if d] + [label]
                nester.prev_leadin = text.endswith(":")

            heading = (
                " > ".join([head, *(f"{lab} {li}".strip() for lab, li in ancestors)])
                if head
                else ""
            )
            uid = make_unit_id(self.name, doc_id, sec_label, *levels)
            n = 2
            while uid in seen:
                uid = make_unit_id(self.name, doc_id, sec_label, *levels) + f"~{n}"
                n += 1
            seen.add(uid)
            out.append(
                Unit(
                    unit_id=uid,
                    source=self.name,
                    doc_id=doc_id,
                    path=[*base_path, *levels],
                    heading=heading,
                    text=text,
                    language="en",
                    version=version,
                    last_amended=last,
                )
            )
        return out
