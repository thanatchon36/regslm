"""Inference for evaluation (docs/specs/train.md): merged checkpoint -> vLLM, greedy, unguided.

Runs in ``.venv`` (vLLM); this module itself only imports light dependencies at top level.
"""

from __future__ import annotations

import glob
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from regslm.corpus.base import Unit
from regslm.train.config import RUNS_DIR, TrainConfig
from regslm.train.data import fixture_split, hash_order, read_label_rows
from regslm.train.prompts import student_messages

SET_NAMES = (
    "test",
    "test_perturbed",
    "heldout",
    "postcutoff",
    "silver_val",
    "gold",
    "thai_test",
    "quick",
)
SPLIT_SETS = {"test", "heldout", "silver_val", "postcutoff", "thai_test"}
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
    elif name == "test_perturbed":
        p = Path("data/local/test_perturbed.jsonl")
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (run `regslm perturb-test`)")
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        items = [
            Item(Unit.model_validate(r["unit"]), r["label"], r["memorisation_targets"])
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


def predict(
    run_id: str,
    sets: list[str],
    limit: int | None = None,
    refs: list[str] | None = None,
    runs: Path = RUNS_DIR,
    gpu_memory_utilization: float | None = None,
) -> dict[str, Any]:
    from regslm.teacher.engine import CHAT_TEMPLATE_KWARGS, EngineConfig, build_llm

    cfg = TrainConfig.load(run_id, runs)
    todo = {n: load_set(n, cfg, refs or [REF_GLOB], limit) for n in sets}
    run_dir = cfg.run_dir(runs)
    out_dir = run_dir / "predictions"
    out_dir.mkdir(exist_ok=True)
    zs = run_dir / "zero_shot.yaml"  # zero-shot cell: base model, student prompt, no checkpoint
    if zs.exists():
        import yaml

        z = yaml.safe_load(zs.read_text())
        model, revision = str(z["model"]), z.get("revision")
    else:
        model, revision = str(model_dir(cfg, runs)), None
    ecfg = EngineConfig(
        model=model, revision=revision, max_model_len=cfg.max_seq_len + MAX_NEW_TOKENS
    )
    if gpu_memory_utilization:
        ecfg.gpu_memory_utilization = gpu_memory_utilization
    from vllm import SamplingParams

    llm = build_llm(ecfg)
    sampling = SamplingParams(
        temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=cfg.seed
    )  # no guided decoding
    meta: dict[str, Any] = {"model": ecfg.model, "sets": {}}
    for name, items in todo.items():
        if not items:
            meta["sets"][name] = {"n": 0}
            continue
        t0 = time.time()
        outs = llm.chat(
            [student_messages(it.unit) for it in items],
            sampling,
            chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        )
        wall = time.time() - t0
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for it, o in zip(items, outs, strict=True):
                c = o.outputs[0]
                m = getattr(o, "metrics", None)
                fin, arr = getattr(m, "finished_time", None), getattr(m, "arrival_time", None)
                lat = fin - arr if fin and arr else wall / len(items)
                row = {
                    "set": name,
                    "unit_id": it.unit.unit_id,
                    "completion": c.text,
                    **decode_row(c.text, it.unit),
                    "n_gen_tokens": len(c.token_ids),
                    "latency_s": lat,
                    "reference": it.reference,
                    "memorisation_targets": it.memorisation_targets,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta["sets"][name] = {"n": len(items), "wall_s": wall}
    (run_dir / "predict_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
