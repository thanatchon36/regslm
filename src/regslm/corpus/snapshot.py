"""Corpus pipeline logic: fetch / parse / snapshot (paper §3.1). CLI wrappers live in cli.py."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
import yaml

from regslm.corpus.base import Document, Licence, RawDoc, SourceAdapter, Unit, write_jsonl


@dataclass(frozen=True)
class Paths:
    """All filesystem roots, configurable for tests."""

    raw: Path = Path("data/raw")
    local: Path = Path("data/local")
    corpus: Path = Path("data/corpus")
    config: Path = Path("config/corpus.yaml")

    @property
    def units_dir(self) -> Path:
        return self.local / "units"

    @property
    def documents_dir(self) -> Path:
        return self.local / "documents"

    @property
    def local_corpus(self) -> Path:
        return self.local / "corpus"


log = logging.getLogger(__name__)


def tier_of(licence: Licence, name: str, excluded: Iterable[str] = ()) -> str:
    if name in set(excluded):
        return "X"
    return "P" if licence.redistributable and licence.derivative_ok else "L"


def load_excluded(config: Path) -> list[str]:
    if not config.exists():
        return []
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return [str(s) for s in data.get("excluded_sources", [])]


def _make(cls: type[SourceAdapter], paths: Paths) -> SourceAdapter:
    return cls(raw_root=paths.raw)


def default_sources(adapters: dict[str, type[SourceAdapter]], excluded: Iterable[str]) -> list[str]:
    """Tier-P adapters, in registry order."""
    return [n for n, c in adapters.items() if tier_of(c.licence, n, excluded) == "P"]


def resolve_sources(
    adapters: dict[str, type[SourceAdapter]], names: list[str] | None, excluded: Iterable[str]
) -> list[str]:
    if not names:
        return default_sources(adapters, excluded)
    unknown = [n for n in names if n not in adapters]
    if unknown:
        raise KeyError(f"unknown source(s) {unknown}; choose from {list(adapters)}")
    return list(names)


def fetch_sources(
    adapters: dict[str, type[SourceAdapter]], names: list[str], paths: Paths
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = sum(1 for _ in _make(adapters[n], paths).fetch())
    return counts


def _parse_adapter(adapter: SourceAdapter) -> tuple[list[Document], list[Unit]]:
    docs: list[Document] = []
    units: list[Unit] = []
    parse_all: Callable[..., list[tuple[Document, list[Unit]]]] | None = getattr(
        adapter, "parse_all", None
    )
    for raw in adapter.fetch():  # cached -> offline
        try:
            pairs = parse_all(raw) if parse_all is not None else [adapter.parse(raw)]
        except Exception as e:
            log.warning("skipped %s (%s): %s", raw.url, type(e).__name__, e)
            continue
        for d, us in pairs:
            docs.append(d)
            units.extend(us)
    return docs, units


def parse_sources(
    adapters: dict[str, type[SourceAdapter]], names: list[str], paths: Paths
) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for n in names:
        docs, units = _parse_adapter(_make(adapters[n], paths))
        write_jsonl(paths.documents_dir / f"{n}.jsonl", list(docs))
        write_jsonl(paths.units_dir / f"{n}.jsonl", list(units))
        out[n] = (len(docs), len(units))
    return out


_M = TypeVar("_M", Document, Unit)


def _read_jsonl(path: Path, model: type[_M]) -> list[_M]:
    return [
        model.model_validate_json(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _raw_docs(raw_root: Path, source: str) -> list[RawDoc]:
    d = raw_root / source
    if not d.exists():
        return []
    out: list[RawDoc] = []
    for m in sorted(d.glob("*.meta.json")):
        raw = RawDoc.model_validate_json(m.read_text(encoding="utf-8"))
        if raw.source == source:
            out.append(raw)
    return out


def _quantile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


@dataclass
class SnapshotResult:
    version: str
    n_documents: int = 0
    n_units_public: int = 0
    n_units_local: int = 0
    files: list[Path] = field(default_factory=list)


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=False)


def build_snapshot(
    adapters: dict[str, type[SourceAdapter]],
    paths: Paths,
    today: date | None = None,
) -> SnapshotResult:
    excluded = load_excluded(paths.config)
    tiers = {n: tier_of(c.licence, n, excluded) for n, c in adapters.items()}
    manifest_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []

    for n in sorted(adapters):
        tier = tiers[n]
        if tier == "X":
            continue
        up, dp = paths.units_dir / f"{n}.jsonl", paths.documents_dir / f"{n}.jsonl"
        if not up.exists() or not dp.exists():
            continue
        docs = _read_jsonl(dp, Document)
        units = _read_jsonl(up, Unit)
        by_sha: dict[str, list[str]] = {}
        for d in docs:
            by_sha.setdefault(d.sha256, []).append(d.doc_id)
            doc_rows.append({**d.model_dump(mode="json"), "tier": tier})
        for u in units:
            unit_rows.append(
                {
                    **u.model_dump(mode="json"),
                    "tier": tier,
                    "n_chars": len(u.text),
                    "text_sha256": hashlib.sha256(u.text.encode("utf-8")).hexdigest(),
                }
            )
        for r in _raw_docs(paths.raw, n):
            manifest_rows.append(
                {
                    "source": n,
                    "url": r.url,
                    "sha256": r.sha256,
                    "fetched_at": r.fetched_at.isoformat(),
                    "content_type": r.content_type,
                    "tier": tier,
                    "doc_ids": sorted(by_sha.get(r.sha256, [])),
                }
            )

    manifest_rows.sort(key=lambda r: (r["source"], r["url"], r["sha256"]))
    doc_rows.sort(key=lambda r: (r["source"], r["doc_id"], r["language"]))
    unit_rows.sort(key=lambda r: r["unit_id"])

    man_schema: dict[str, Any] = {
        "source": pl.String,
        "url": pl.String,
        "sha256": pl.String,
        "fetched_at": pl.String,
        "content_type": pl.String,
        "tier": pl.String,
        "doc_ids": pl.List(pl.String),
    }
    manifest = pl.DataFrame(manifest_rows, schema=man_schema)
    doc_schema: dict[str, Any] = {k: pl.String for k in (*Document.model_fields, "tier")}
    documents = pl.DataFrame(
        doc_rows,
        schema=doc_schema,
    )
    unit_schema: dict[str, Any] = {k: pl.String for k in Unit.model_fields}
    unit_schema["path"] = pl.List(pl.String)
    unit_schema.update(tier=pl.String, n_chars=pl.Int64, text_sha256=pl.String)
    units_all = pl.DataFrame(unit_rows, schema=unit_schema)
    public = units_all.filter(pl.col("tier") == "P")
    local = units_all.filter(pl.col("tier") == "L")

    digest = hashlib.sha256("\n".join(sorted(public["text_sha256"].to_list())).encode()).hexdigest()
    version = f"{(today or date.today()).strftime('%Y.%m.%d')}-{digest[:8]}"

    # Only tier-P document rows are published; L documents (titles/urls) stay local with L units.
    pub_docs = documents.filter(pl.col("tier") == "P")
    out = paths.corpus
    res = SnapshotResult(
        version=version,
        n_documents=pub_docs.height,
        n_units_public=public.height,
        n_units_local=local.height,
    )
    _write_parquet(manifest, out / "manifest.parquet")
    _write_parquet(pub_docs, out / "documents.parquet")
    _write_parquet(public, out / "units.parquet")
    res.files = [out / "manifest.parquet", out / "documents.parquet", out / "units.parquet"]
    local_path = paths.local_corpus / "units_local.parquet"
    if local.height:
        _write_parquet(local, local_path)
        _write_parquet(
            documents.filter(pl.col("tier") == "L"), paths.local_corpus / "documents_local.parquet"
        )
    else:
        local_path.unlink(missing_ok=True)
        (paths.local_corpus / "documents_local.parquet").unlink(missing_ok=True)
    (out / "VERSION").write_text(version + "\n", encoding="utf-8")
    (out / "STATS.md").write_text(render_stats(version, pub_docs, public), encoding="utf-8")
    return res


def render_stats(version: str, docs: pl.DataFrame, units: pl.DataFrame) -> str:
    lines = [
        "# Corpus statistics",
        "",
        f"Version: `{version}` (tier-P only)",
        "",
        "| source | doc | language | docs | units | chars median | chars p95 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    keys = sorted({(r["source"], r["doc_id"], r["language"]) for r in docs.iter_rows(named=True)})
    for s, d, lg in keys:
        sel = units.filter((pl.col("source") == s) & (pl.col("doc_id") == d))
        sel = sel.filter(pl.col("language") == lg)
        ch = sorted(sel["n_chars"].to_list())
        lines.append(
            f"| {s} | {d} | {lg} | 1 | {len(ch)} | {_quantile(ch, 0.5):.0f} "
            f"| {_quantile(ch, 0.95):.0f} |"
        )
    lines.append("")
    lines.append(f"Total: {docs.height} documents, {units.height} units.")
    lines.append("")
    return "\n".join(lines)
