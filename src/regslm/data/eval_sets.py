"""Materialise the reference test sets (docs/RESEARCH_QUESTIONS.md section 0, D-034/D-037).

Each set is ``data/local/eval_sets/<name>.jsonl`` with rows ``{unit, reference, changes?,
memorisation_targets?}``; perturbed rows carry the edit list and the memorisation targets
(threshold and deadline values, see ``perturb_set``). ``MANIFEST.json`` holds counts and the
sha256 of the labels file the references came from.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from regslm.corpus.base import Unit
from regslm.data.perturb_set import perturb_unit_and_label, read_labels
from regslm.schema.obligation import Extraction

EVAL_SETS_DIR = Path("data/local/eval_sets")
POOL_SPLITS = ("test", "dev", "silver_val", "heldout")
SET_ORDER = (
    "test", "test_perturbed", "num440", "num440_perturbed", "mr_pool_s0", "mr_pool_s1",
    "heldout", "silver_val", "silver_val_1k", "postcutoff", "thai_test",
)  # fmt: skip


def _row(
    unit: Unit, ref: Extraction, changes: Any = None, targets: list[Any] | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "unit": unit.model_dump(mode="json"),
        "reference": ref.model_dump(mode="json"),
    }
    if changes is not None:
        row["changes"] = {
            "text": [vars(c) for c in changes.text],
            "heading": [vars(c) for c in changes.heading],
        }
        row["memorisation_targets"] = targets or []
    return row


def build_sets(
    units: Mapping[str, Unit],
    labels: Mapping[str, Extraction],
    assignments: list[dict[str, Any]],
    gold: Mapping[str, Extraction] | None = None,
    sample_n: int = 1000,
) -> dict[str, list[dict[str, Any]]]:
    """Pure construction of every set. ``assignments`` rows: unit_id, split, language,
    postcutoff. Units without a unit record or a valid reference are skipped. ``gold``
    (finalised gold references) joins the memorisation pool when given."""
    from regslm.train.data import hash_order

    split_of = {a["unit_id"]: a for a in assignments}
    refs: dict[str, Extraction] = dict(labels)
    for uid, g in (gold or {}).items():
        refs[uid] = g

    def ids(*splits: str) -> list[str]:
        return sorted(
            u for u, a in split_of.items() if a["split"] in splits and u in units and u in refs
        )

    def plain(uids: list[str]) -> list[dict[str, Any]]:
        return [_row(units[u], refs[u]) for u in uids]

    def perturbed(uids: list[str], seed: int, only_changed: bool) -> list[dict[str, Any]]:
        out = []
        for u in uids:
            nu, nex, ch = perturb_unit_and_label(units[u], refs[u], seed)
            assert nex is not None
            if only_changed and not ch.all:
                continue
            out.append(_row(nu, nex, ch, [t.as_row() for t in ch.targets]))
        return out

    test = ids("test")
    changed = [u for u in test if perturb_unit_and_label(units[u], None, 0)[2].all]
    pool = [
        u
        for u in ids(*POOL_SPLITS, *(["gold"] if gold else []))
        if perturb_unit_and_label(units[u], None, 0)[2].all
    ]
    sv = ids("silver_val")
    sv1k = set(hash_order(sv, 0)[:sample_n])
    return {
        "test": plain(test),
        "test_perturbed": perturbed(test, 0, False),
        "num440": plain(changed),
        "num440_perturbed": perturbed(changed, 0, False),
        "mr_pool_s0": perturbed(pool, 0, True),
        "mr_pool_s1": perturbed(pool, 1, True),
        "heldout": plain(ids("heldout")),
        "silver_val": plain(sv),
        "silver_val_1k": plain([u for u in sv if u in sv1k]),
        "postcutoff": plain([u for u in test if split_of[u]["postcutoff"]]),
        "thai_test": plain([u for u in test if split_of[u]["language"] == "th"]),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sets(
    sets: Mapping[str, list[dict[str, Any]]], out: Path, labels_path: Path
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "labels": str(labels_path),
        "labels_sha256": file_sha256(labels_path),
        "sets": {},
    }
    for name, rows in sets.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["sets"][name] = {
            "n": len(rows),
            "n_targets": sum(len(r.get("memorisation_targets", [])) for r in rows),
        }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_set(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_gold(path: Path) -> dict[str, Extraction]:
    """Finalised gold references (``{"unit_id", "gold": Extraction}`` rows); {} if absent."""
    if not path.exists():
        return {}
    out: dict[str, Extraction] = {}
    for r in read_set(path):
        out[r["unit_id"]] = Extraction.model_validate(r["gold"])
    return out


def build_from_files(
    labels_path: Path,
    out: Path = EVAL_SETS_DIR,
    assignments_path: Path = Path("data/splits/assignments.parquet"),
    gold_path: Path = Path("data/gold/gold.jsonl"),
) -> dict[str, Any]:
    import polars as pl

    assign = pl.read_parquet(assignments_path)
    frames = [pl.read_parquet("data/corpus/units.parquet")]
    local = Path("data/local/corpus/units_local.parquet")
    if local.exists():
        frames.append(pl.read_parquet(local))
    df = pl.concat(frames, how="diagonal").unique("unit_id")
    df = df.filter(pl.col("unit_id").is_in(assign["unit_id"].to_list()))
    units = {
        r["unit_id"]: Unit.model_validate(r)
        for r in df.select(list(Unit.model_fields)).iter_rows(named=True)
    }
    sets = build_sets(
        units,
        read_labels(labels_path),
        assign.select(["unit_id", "split", "language", "postcutoff"]).to_dicts(),
        load_gold(gold_path) or None,
    )
    return write_sets(sets, out, labels_path)
