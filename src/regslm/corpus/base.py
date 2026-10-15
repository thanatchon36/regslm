"""Corpus contracts shared by every source adapter (paper §3.1 "Corpus").

An adapter turns one regulator's public website / API into:

- ``RawDoc``    — one fetched response, cached on disk under ``data/raw/<source>/``
                  with URL, SHA-256, fetch time, content type.
- ``Document``  — one legal instrument / chapter (e.g. CRR, 12 CFR Part 217).
- ``Unit``      — one hierarchically-addressed paragraph-level text unit; the
                  thing that is labelled. ``unit_id`` is
                  ``"<source>/<doc>/<part>/<section>/<paragraph>"`` with empty
                  levels dropped, and is stable across re-fetches of the same
                  version.

Polite-fetching rules (MISSION §0) are enforced by ``PoliteClient``, not by each
adapter: robots.txt honoured, >= 2 s between requests per host, descriptive UA,
every response cached by URL so re-runs are offline.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

USER_AGENT = (
    "regslm/0.1 (open-source research; public regulatory text; "
    "+https://github.com/thanatchon36/regslm)"
)
MIN_INTERVAL_S = 2.0


class Licence(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    url: str
    redistributable: bool  # may the text itself be committed / released?
    derivative_ok: bool  # may released model weights be trained on it?
    attribution: str = ""


class RawDoc(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    url: str
    sha256: str
    path: str  # relative to repo root
    content_type: str
    fetched_at: datetime


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    doc_id: str
    title: str
    language: str  # BCP-47, e.g. "en", "th"
    url: str
    version: str  # point-in-time / consolidation date, ISO-8601
    sha256: str  # of the raw response it was parsed from


class Unit(BaseModel):
    model_config = ConfigDict(frozen=True)

    unit_id: str
    source: str
    doc_id: str
    # hierarchy labels, e.g. ['Part 217', '§ 217.10', '(a)', '(1)']
    path: list[str]
    heading: str = ""
    text: str
    language: str
    version: str
    last_amended: str | None = None  # ISO date if the source exposes it (post-cutoff slice)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_unit_id(source: str, doc_id: str, *levels: str) -> str:
    clean = [s.strip().replace("/", "_").replace(" ", "") for s in (source, doc_id, *levels)]
    return "/".join(p for p in clean if p)


@dataclass
class PoliteClient:
    """HTTP client with on-disk cache, robots.txt check and per-host rate limit."""

    cache_dir: Path
    source: str
    min_interval_s: float = MIN_INTERVAL_S
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=self.timeout_s, follow_redirects=True
        )

    def _key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self._http.get(f"{host}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self._robots[host] = rp
        return self._robots[host].can_fetch(USER_AGENT, url)

    def _wait(self, url: str) -> None:
        host = urlsplit(url).netloc
        dt = time.monotonic() - self._last.get(host, 0.0)
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last[host] = time.monotonic()

    def get(self, url: str, suffix: str = "") -> RawDoc:
        """Return the cached response for ``url``, fetching it once if needed.

        Raises ``PermissionError`` if robots.txt disallows the URL and
        ``httpx.HTTPStatusError`` on non-2xx responses (adapters turn persistent
        403/429 into an entry in docs/sources/NEEDED.md).
        """
        key = self._key(url)
        body = self.cache_dir / f"{key}{suffix}"
        meta = self.cache_dir / f"{key}.meta.json"
        if body.exists() and meta.exists():
            return RawDoc.model_validate_json(meta.read_text())
        if not self._allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        self._wait(url)
        r = self._http.get(url)
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

    def read(self, raw: RawDoc) -> bytes:
        return Path(raw.path).read_bytes()


class SourceAdapter(ABC):
    """One per regulator source. Subclasses document: document types, URL patterns,
    structure, language(s), and licence terms (mirrored in docs/sources/LICENSES.md)."""

    name: str
    licence: Licence

    def __init__(self, raw_root: Path = Path("data/raw")) -> None:
        self.client = PoliteClient(cache_dir=raw_root / self.name, source=self.name)

    @abstractmethod
    def fetch(self, selection: list[str] | None = None) -> Iterator[RawDoc]:
        """Download (or load from cache) the raw responses for the selected documents."""

    @abstractmethod
    def parse(self, raw: RawDoc) -> tuple[Document, list[Unit]]:
        """Parse one raw response into a Document and its ordered Units."""


def write_jsonl(path: Path, rows: Iterator[BaseModel] | list[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
            n += 1
    return n
