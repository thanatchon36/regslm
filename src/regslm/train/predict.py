"""Inference for evaluation (docs/specs/train.md): merged checkpoint -> vLLM, greedy, unguided.

Runs in ``.venv`` (vLLM); this module itself only imports light dependencies at top level.
"""

from __future__ import annotations

import glob
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regslm.corpus.base import Unit
from regslm.data.eval_sets import EVAL_SETS_DIR
from regslm.train.config import RUNS_DIR, TrainConfig
from regslm.train.data import fixture_split, hash_order, read_label_rows
from regslm.train.prompts import student_messages, zero_shot_messages

SET_NAMES = (
    "test",
    "test_perturbed",
    "num440",
    "num440_perturbed",
    "mr_pool_s0",
    "mr_pool_s1",
    "heldout",
    "postcutoff",
    "silver_val",
    "silver_val_1k",
    "gold",
    "thai_test",
    "quick",
)
SPLIT_SETS = {"test", "heldout", "silver_val", "postcutoff", "thai_test"}
# Materialised by `regslm eval-sets` (data/local/eval_sets/<name>.jsonl); preferred over the
# legacy split path when present so that every set shares one reference-label source.
FILE_SETS = {
    "test_perturbed", "num440", "num440_perturbed", "mr_pool_s0", "mr_pool_s1", "silver_val_1k",
}  # fmt: skip
# Sets whose predictions are reused from another set wherever the prompt text is identical
# (D-034): unchanged units of test_perturbed, the pure subsets of test / test_perturbed, and the
# test units of mr_pool_s0 (same seed and text as test_perturbed).
REUSE_FROM = {
    "test_perturbed": "test",
    "num440": "test",
    "num440_perturbed": "test_perturbed",
    "mr_pool_s0": "test_perturbed",  # seed 0 = same perturbation as test_perturbed
    "postcutoff": "test",
    "thai_test": "test",
}
REF_GLOB = "results/runs/silver-*/labels.jsonl"
MAX_NEW_TOKENS = 1024


@dataclass
class Item:
    unit: Unit
    reference: dict[str, Any] | None
    memorisation_targets: list[Any] = field(default_factory=list)


def _reference_map(patterns: list[str]) -> dict[str, dict[str, Any]]:
    """unit_id -> valid reference Extraction dict, from teacher labels.jsonl files."""
    out: dict[str, dict[str, Any]] = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for uid, r in read_label_rows([path]).items():
                if r.valid and r.extraction is not None:
                    out[uid] = r.extraction.model_dump(mode="json")
    return out


def _units_by_id(ids: set[str]) -> dict[str, Unit]:
    import polars as pl

    frames = [pl.read_parquet("data/corpus/units.parquet")]
    local = Path("data/local/corpus/units_local.parquet")
    if local.exists():
        frames.append(pl.read_parquet(local))
    df = pl.concat(frames, how="diagonal").unique("unit_id")
    df = df.filter(pl.col("unit_id").is_in(sorted(ids)))
    return {
        r["unit_id"]: Unit.model_validate(r)
        for r in df.select(list(Unit.model_fields)).iter_rows(named=True)
    }


def load_set(name: str, cfg: TrainConfig, refs: list[str], limit: int | None = None) -> list[Item]:
    """Items for one test set. Units without a reference are skipped (cannot be scored)."""
    items: list[Item]
    if name == "quick":
        _, _, held = fixture_split(read_label_rows(cfg.labels), cfg.gold_sample, cfg.eval_n)
        items = [
            Item(r.unit, r.extraction.model_dump(mode="json") if r.extraction else None)
            for r in held
            if r.unit is not None
        ]
    elif name in FILE_SETS or (name in SPLIT_SETS and (EVAL_SETS_DIR / f"{name}.jsonl").exists()):
        p = EVAL_SETS_DIR / f"{name}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (run `regslm eval-sets --labels ...`)")
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        items = [
            Item(Unit.model_validate(r["unit"]), r["reference"], r.get("memorisation_targets", []))
            for r in rows
        ]
    elif name == "gold":
        p = Path("data/gold/gold.jsonl")
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (gold not finalised)")
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        units = _units_by_id({r["unit_id"] for r in rows})
        items = [Item(units[r["unit_id"]], r["gold"]) for r in rows if r["unit_id"] in units]
    elif name in SPLIT_SETS:
        import polars as pl

        a = pl.read_parquet("data/splits/assignments.parquet")
        if name == "test":
            a = a.filter(pl.col("split") == "test")
        elif name == "thai_test":
            a = a.filter((pl.col("split") == "test") & (pl.col("language") == "th"))
        elif name == "postcutoff":
            a = a.filter((pl.col("split") == "test") & pl.col("postcutoff"))
        else:
            a = a.filter(pl.col("split") == name)
        refmap = _reference_map(refs)
        ids = [i for i in a["unit_id"].to_list() if i in refmap]
        units = _units_by_id(set(ids))
        items = [Item(units[i], refmap[i]) for i in ids if i in units]
    else:
        raise ValueError(f"unknown test set {name!r}; choose from {SET_NAMES}")
    order = {u: i for i, u in enumerate(hash_order(it.unit.unit_id for it in items))}
    items.sort(key=lambda it: order[it.unit.unit_id])
    return items[:limit] if limit is not None else items


def model_dir(cfg: TrainConfig, runs: Path = RUNS_DIR) -> Path:
    d = cfg.run_dir(runs) / "checkpoints" / "merged"
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"no merged checkpoint at {d}; train first")
    return d


def decode_row(completion: str, unit: Unit) -> dict[str, Any]:
    """Parse + resolve one completion (no repair)."""
    from regslm.rewards.components import resolved

    r = resolved(completion, unit)
    if r is None:
        return {"valid": False, "extraction": None, "n_unresolved": 0, "wrong_unit": 0}
    return {
        "valid": True,
        "extraction": r.extraction.model_dump(mode="json"),
        "n_unresolved": len(r.unresolved),
        "wrong_unit": r.wrong_unit,
    }


def order_sets(names: list[str]) -> list[str]:
    """De-duplicated, with each set after the set its predictions are reused from."""
    out: list[str] = []

    def add(n: str) -> None:
        if n in out:
            return
        src = REUSE_FROM.get(n)
        if src in names:
            add(src)
        out.append(n)

    for n in names:
        add(n)
    return out


def _same_prompt(a: Unit, b: Unit) -> bool:
    return a.text == b.text and a.heading == b.heading


def assemble_set(
    name: str,
    items: list[Item],
    base_items: dict[str, Item],
    base_rows: dict[str, dict[str, Any]],
    generate: Callable[[list[Item]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    """Prediction rows for ``items`` (same order) and the number of units actually generated.

    A unit is not re-run when ``base_rows`` (predictions of ``REUSE_FROM[name]``) hold a row
    for it and the base item has the identical prompt text; the row is copied with this set's
    reference and targets. ``generate`` maps items to partial rows (completion, decode fields,
    n_gen_tokens, latency_s).
    """
    reuse: dict[str, dict[str, Any]] = {}
    for it in items:
        b = base_items.get(it.unit.unit_id)
        r = base_rows.get(it.unit.unit_id)
        if b is not None and r is not None and _same_prompt(b.unit, it.unit):
            reuse[it.unit.unit_id] = r
    run = [it for it in items if it.unit.unit_id not in reuse]
    fresh = iter(generate(run) if run else [])
    rows: list[dict[str, Any]] = []
    for it in items:
        r = reuse.get(it.unit.unit_id)
        part = dict(r) if r is not None else next(fresh)
        part.update(
            set=name,
            unit_id=it.unit.unit_id,
            reference=it.reference,
            memorisation_targets=it.memorisation_targets,
        )
        rows.append(part)
    return rows, len(run)


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {r["unit_id"]: r for r in rows}


def predict(
    run_id: str,
    sets: list[str],
    limit: int | None = None,
    refs: list[str] | None = None,
    runs: Path = RUNS_DIR,
    gpu_memory_utilization: float | None = None,
) -> dict[str, Any]:
    cfg = TrainConfig.load(run_id, runs)
    run_dir = cfg.run_dir(runs)
    out_dir = run_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    names = order_sets(sets)
    pending: list[str] = []
    if "gold" in names and not Path("data/gold/gold.jsonl").exists():
        # Gold awaits owner adjudication: record as pending, predict it later (D-044).
        names = [n for n in names if n != "gold"]
        pending.append("gold")
    todo = {n: load_set(n, cfg, refs or [REF_GLOB], limit) for n in names}
    llm_box: list[Any] = []

    def generate(items: list[Item]) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        from regslm.teacher.engine import CHAT_TEMPLATE_KWARGS, EngineConfig, build_llm

        if not llm_box:
            zs = run_dir / "zero_shot.yaml"  # zero-shot cell: base model, teacher prompt (D-043)
            max_len = cfg.max_seq_len + MAX_NEW_TOKENS
            if zs.exists():
                import yaml

                z = yaml.safe_load(zs.read_text())
                model, revision = str(z["model"]), z.get("revision")
                max_len = 4096 + MAX_NEW_TOKENS  # long teacher prompt + schema
            else:
                model, revision = str(model_dir(cfg, runs)), None
            ecfg = EngineConfig(model=model, revision=revision, max_model_len=max_len)
            if gpu_memory_utilization:
                ecfg.gpu_memory_utilization = gpu_memory_utilization
            llm_box.append((build_llm(ecfg), ecfg.model))
        llm = llm_box[0][0]
        sampling = SamplingParams(
            temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=cfg.seed
        )  # no guided decoding
        t0 = time.time()
        build = zero_shot_messages if (run_dir / "zero_shot.yaml").exists() else student_messages
        outs = llm.chat(
            [build(it.unit) for it in items],
            sampling,
            chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        )
        wall = time.time() - t0
        rows = []
        for it, o in zip(items, outs, strict=True):
            c = o.outputs[0]
            m = getattr(o, "metrics", None)
            fin, arr = getattr(m, "finished_time", None), getattr(m, "arrival_time", None)
            rows.append(
                {
                    "completion": c.text,
                    **decode_row(c.text, it.unit),
                    "n_gen_tokens": len(c.token_ids),
                    "latency_s": fin - arr if fin and arr else wall / len(items),
                }
            )
        return rows

    meta: dict[str, Any] = {"sets": {}, "pending_sets": pending}
    done_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for name in names:
        items = todo[name]
        if not items:
            meta["sets"][name] = {"n": 0, "n_units": 0, "n_reused": 0}
            continue
        src = REUSE_FROM.get(name)
        base_items: dict[str, Item] = {}
        base_rows: dict[str, dict[str, Any]] = {}
        if src is not None:
            base_rows = done_rows.get(src) or _read_rows(out_dir / f"{src}.jsonl")
            if base_rows:
                base = load_set(src, cfg, refs or [REF_GLOB], limit)
                base_items = {it.unit.unit_id: it for it in base}
        t0 = time.time()
        rows, n_run = assemble_set(name, items, base_items, base_rows, generate)
        wall = time.time() - t0
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done_rows[name] = {r["unit_id"]: r for r in rows}
        # "n" = units actually generated (experiments/throughput.py divides n by wall_s)
        meta["sets"][name] = {
            "n": n_run, "n_units": len(items), "n_reused": len(items) - n_run, "wall_s": wall,
        }  # fmt: skip
    if llm_box:
        meta["model"] = llm_box[0][1]
    (run_dir / "predict_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
