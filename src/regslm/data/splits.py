"""Grouped splits (paper §3.5; D-023, D-024, D-025). Output holds ids only, never text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

SPLITS = ("train", "dev", "test", "silver_val", "heldout", "gold")
TOLERANCE = 0.01  # ±1 pp


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    heldout_docs: tuple[str, ...] = ("eurlex/AMLD4",)
    test_frac: float = 0.10
    dev_frac: float = 0.05
    silver_val_n: int = 2000
    postcutoff_from: str = "2025-05-01"
    group_levels: int = 2
    max_group_frac: float = 0.01  # groups above this share of the pool are refined deeper


def load_config(path: Path = Path("config/splits.yaml")) -> SplitConfig:
    d: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    d["heldout_docs"] = tuple(d.get("heldout_docs", ()))
    d["postcutoff_from"] = str(d.get("postcutoff_from", "2025-05-01"))
    return SplitConfig(**d)


def _h(seed: int, salt: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{key}".encode()).hexdigest()


def read_gold_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            ids.add(str(json.loads(ln)["unit_id"]))
    return ids


def load_units(corpus: Path, local: Path | None) -> pl.DataFrame:
    """Unit columns needed for splitting (no text) from public (+ local) parquet."""
    cols = ["unit_id", "source", "doc_id", "language", "path", "last_amended"]
    frames = [pl.read_parquet(corpus).select(cols)]
    if local is not None and local.exists():
        frames.append(pl.read_parquet(local).select(cols))
    return pl.concat(frames).unique("unit_id").sort("unit_id")


def _fill(groups: list[tuple[str, int]], target: int, taken: dict[str, str], label: str) -> None:
    """Assign whole groups (already hash-ordered) until the unit count is closest to target."""
    total = 0
    for g, n in groups:
        if g in taken:
            continue
        if total >= target:
            break
        if abs(total + n - target) >= abs(total - target):
            continue  # overshoots more than it helps; try a smaller group
        taken[g] = label
        total += n


def _group_ids(units: pl.DataFrame, cfg: SplitConfig) -> list[str]:
    """Group by the first ``group_levels`` path levels; oversized groups (coarse Part/Subpart
    or Title levels in some sources) are refined one level at a time until small enough,
    which keeps the split fractions achievable while still grouping at article/section scale."""
    rows = list(units.select("source", "doc_id", "path").iter_rows())
    limit = max(1, int(cfg.max_group_frac * len(rows)))
    levels = [cfg.group_levels] * len(rows)

    def key(i: int) -> str:
        src, doc, path = rows[i]
        return "/".join([src, doc, *path[: levels[i]]])

    while True:
        keys = [key(i) for i in range(len(rows))]
        counts: dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        grew = False
        for i, k in enumerate(keys):
            if counts[k] > limit and levels[i] < len(rows[i][2]):
                levels[i] += 1
                grew = True
        if not grew:
            return keys


def assign(units: pl.DataFrame, gold_ids: set[str], cfg: SplitConfig) -> pl.DataFrame:
    gids = _group_ids(units, cfg)
    df = units.with_columns(pl.Series("group_id", gids, dtype=pl.String))
    heldout = set(cfg.heldout_docs)
    is_gold = pl.col("unit_id").is_in(sorted(gold_ids))
    is_held = (pl.col("source") + "/" + pl.col("doc_id")).is_in(sorted(heldout))
    df = df.with_columns(
        pl.when(is_gold)
        .then(pl.lit("gold"))
        .when(is_held)
        .then(pl.lit("heldout"))
        .otherwise(pl.lit(""))
        .alias("split")
    )
    pool = df.filter(pl.col("split") == "")
    sizes = pool.group_by("group_id").len()
    ordered = sorted(
        ((g, int(n)) for g, n in sizes.iter_rows()),
        key=lambda x: (_h(cfg.seed, "split", x[0]), x[0]),
    )
    n_pool = pool.height
    taken: dict[str, str] = {}
    _fill(ordered, round(cfg.test_frac * n_pool), taken, "test")
    _fill(ordered, round(cfg.dev_frac * n_pool), taken, "dev")
    for g, _ in ordered:
        taken.setdefault(g, "train")
    # silver_val: whole groups from the train side, independent hash order.
    train_groups = [(g, n) for g, n in ordered if taken[g] == "train"]
    train_groups.sort(key=lambda x: (_h(cfg.seed, "silver", x[0]), x[0]))
    sv: dict[str, str] = {}
    _fill(train_groups, cfg.silver_val_n, sv, "silver_val")
    taken.update(sv)
    split = pl.col("split")
    df = df.with_columns(
        pl.when(split == "")
        .then(pl.col("group_id").replace_strict(taken, default="train", return_dtype=pl.String))
        .otherwise(split)
        .alias("split"),
        (pl.col("last_amended").fill_null("") >= cfg.postcutoff_from)
        .and_(pl.col("last_amended").is_not_null())
        .alias("postcutoff"),
    )
    return df.select(
        "unit_id", "source", "doc_id", "language", "group_id", "split", "postcutoff"
    ).sort("unit_id")


def fraction_deviation(a: pl.DataFrame, cfg: SplitConfig) -> dict[str, float]:
    """Actual - target fraction (of units outside gold/heldout) for test and dev."""
    pool = a.filter(~pl.col("split").is_in(["gold", "heldout"]))
    n = max(pool.height, 1)
    out = {}
    for name, frac in (("test", cfg.test_frac), ("dev", cfg.dev_frac)):
        out[name] = (pool.filter(pl.col("split") == name).height / n) - frac
    return out


def version_hash(a: pl.DataFrame) -> str:
    body = "\n".join(f"{u}\t{s}" for u, s in a.select("unit_id", "split").iter_rows())
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def render_splits_md(a: pl.DataFrame, cfg: SplitConfig, corpus_version: str = "") -> str:
    dev = fraction_deviation(a, cfg)
    ok = all(abs(v) <= TOLERANCE for v in dev.values())
    lines = [
        "# Splits",
        "",
        f"VERSION: `{version_hash(a)}` (seed {cfg.seed}, corpus `{corpus_version or 'n/a'}`)",
        "",
        f"Grouping: first {cfg.group_levels} path levels within a document (D-024), "
        f"refined deeper for groups > {cfg.max_group_frac:.0%} of the pool. "
        f"Held-out docs: {', '.join(cfg.heldout_docs) or 'none'}. "
        f"Post-cutoff: last_amended >= {cfg.postcutoff_from} (D-025).",
        "",
        "Fraction check (actual - target, of train/dev/test/silver pool): "
        + ", ".join(f"{k} {v * 100:+.2f} pp" for k, v in dev.items())
        + (" (within ±1 pp)" if ok else " (**OUTSIDE ±1 pp**)"),
        "",
        "## Units per split x source x language",
        "",
        "| split | source | language | units | postcutoff |",
        "|---|---|---|---:|---:|",
    ]
    g = (
        a.group_by("split", "source", "language")
        .agg(pl.len().alias("n"), pl.col("postcutoff").sum().alias("pc"))
        .to_dicts()
    )
    g.sort(key=lambda r: (SPLITS.index(r["split"]), r["source"], r["language"]))
    for r in g:
        lines.append(f"| {r['split']} | {r['source']} | {r['language']} | {r['n']} | {r['pc']} |")
    lines += ["", "## Totals", "", "| split | units | postcutoff |", "|---|---:|---:|"]
    for s in SPLITS:
        sel = a.filter(pl.col("split") == s)
        lines.append(f"| {s} | {sel.height} | {int(sel['postcutoff'].sum() or 0)} |")
    lines.append(f"| all | {a.height} | {int(a['postcutoff'].sum() or 0)} |")
    lines.append("")
    return "\n".join(lines)


def build_splits(
    cfg: SplitConfig,
    corpus: Path = Path("data/corpus/units.parquet"),
    local: Path | None = Path("data/local/corpus/units_local.parquet"),
    gold: Path = Path("data/gold/sample.jsonl"),
    out: Path = Path("data/splits"),
) -> pl.DataFrame:
    a = assign(load_units(corpus, local), read_gold_ids(gold), cfg)
    out.mkdir(parents=True, exist_ok=True)
    a.write_parquet(out / "assignments.parquet", compression="zstd", statistics=False)
    vfile = corpus.parent / "VERSION"
    cv = vfile.read_text(encoding="utf-8").strip() if vfile.exists() else ""
    (out / "SPLITS.md").write_text(render_splits_md(a, cfg, cv), encoding="utf-8")
    return a
