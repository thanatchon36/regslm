from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from regslm.corpus import snapshot as sn
from regslm.corpus.base import Document, Licence, RawDoc, SourceAdapter, Unit

P_LIC = Licence(name="pd", url="u", redistributable=True, derivative_ok=True)
L_LIC = Licence(name="nc", url="u", redistributable=True, derivative_ok=False)


def _adapter(name: str, lic: Licence) -> type[SourceAdapter]:
    class A(SourceAdapter):
        licence = lic

        def fetch(self, selection: list[str] | None = None) -> Iterator[RawDoc]:
            yield RawDoc(
                source=name,
                url=f"http://x/{name}",
                sha256=f"sha-{name}",
                path="p",
                content_type="text/plain",
                fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

        def parse(self, raw: RawDoc) -> tuple[Document, list[Unit]]:
            d = Document(
                source=name,
                doc_id="D",
                title="t",
                language="en",
                url=raw.url,
                version="2026-01-01",
                sha256=raw.sha256,
            )
            us = [
                Unit(
                    unit_id=f"{name}/D/{i}",
                    source=name,
                    doc_id="D",
                    path=[str(i)],
                    text=f"SECRET-{name}-{i}" * (i + 1),
                    language="en",
                    version="2026-01-01",
                )
                for i in range(3)
            ]
            return d, us

    A.name = name
    return A


def _setup(tmp_path: Path) -> tuple[dict[str, type[SourceAdapter]], sn.Paths]:
    adapters = {"pub": _adapter("pub", P_LIC), "loc": _adapter("loc", L_LIC)}
    paths = sn.Paths(
        raw=tmp_path / "raw",
        local=tmp_path / "local",
        corpus=tmp_path / "corpus",
        config=tmp_path / "none.yaml",
    )
    for n in adapters:
        (paths.raw / n).mkdir(parents=True)
        raw = next(iter(adapters[n](paths.raw).fetch()))
        (paths.raw / n / "k.meta.json").write_text(raw.model_dump_json())
    sn.parse_sources(adapters, list(adapters), paths)
    return adapters, paths


def test_tier_routing(tmp_path: Path) -> None:
    adapters, paths = _setup(tmp_path)
    assert sn.default_sources(adapters, []) == ["pub"]
    assert sn.tier_of(P_LIC, "pub", ["pub"]) == "X"
    res = sn.build_snapshot(adapters, paths)
    pub = pl.read_parquet(paths.corpus / "units.parquet")
    assert set(pub["tier"]) == {"P"} and pub.height == 3
    loc = pl.read_parquet(paths.local_corpus / "units_local.parquet")
    assert set(loc["tier"]) == {"L"} and res.n_units_local == 3
    assert set(pl.read_parquet(paths.corpus / "documents.parquet")["source"]) == {"pub"}
    man = pl.read_parquet(paths.corpus / "manifest.parquet")
    assert set(man["tier"]) == {"P", "L"}
    assert man.filter(pl.col("source") == "pub")["doc_ids"].to_list() == [["D"]]


def test_manifest_has_no_text_and_no_l_text_in_corpus(tmp_path: Path) -> None:
    adapters, paths = _setup(tmp_path)
    sn.build_snapshot(adapters, paths)
    man = pl.read_parquet(paths.corpus / "manifest.parquet")
    assert "text" not in man.columns
    for f in paths.corpus.rglob("*"):
        if f.is_file():
            assert b"SECRET-loc" not in f.read_bytes(), f


def test_determinism(tmp_path: Path) -> None:
    adapters, paths = _setup(tmp_path)
    from datetime import date

    day = date(2026, 3, 4)
    a = sn.build_snapshot(adapters, paths, today=day)
    first = {f.name: f.read_bytes() for f in paths.corpus.iterdir()}
    b = sn.build_snapshot(adapters, paths, today=day)
    second = {f.name: f.read_bytes() for f in paths.corpus.iterdir()}
    assert a.version == b.version and a.version.startswith("2026.03.04-")
    assert first == second
