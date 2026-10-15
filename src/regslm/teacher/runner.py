"""Batch labelling of corpus units with the local teacher (paper §3.4)."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from regslm.corpus.base import Unit
from regslm.schema.generation import GenExtraction, resolve
from regslm.teacher.cache import CacheRow, TeacherCache, cache_key
from regslm.teacher.engine import (
    CHAT_TEMPLATE_KWARGS,
    EngineConfig,
    build_llm,
    build_sampling,
)
from regslm.teacher.prompts import Variant, system_prompt, user_prompt

PROJECTION_SIZES = (30000, 50000)


def build_messages(unit: Unit, variant: Variant) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(variant)},
        {
            "role": "user",
            "content": user_prompt(unit.source, unit.unit_id, unit.heading, unit.text),
        },
    ]


def validate_output(text: str, unit_id: str) -> tuple[str, bool]:
    """Return (stored output, valid). Valid outputs get ``unit_id`` forced to the input's."""
    try:
        ex = GenExtraction.model_validate_json(text)
    except ValueError:
        return text, False
    return ex.model_copy(update={"unit_id": unit_id}).model_dump_json(), True


def _metrics(
    n_done: int,
    n_valid: int,
    new_units: int,
    gen: int,
    prompt: int,
    elapsed: float,
) -> dict[str, Any]:
    uph = new_units / elapsed * 3600 if elapsed > 0 and new_units else 0.0
    return {
        "n_done": n_done,
        "n_valid": n_valid,
        "valid_rate": n_valid / n_done if n_done else 0.0,
        "gen_tokens": gen,
        "prompt_tokens": prompt,
        "tok_per_s": gen / elapsed if elapsed > 0 else 0.0,
        "units_per_hour": uph,
        **{f"projected_hours_for_{n}": (n / uph if uph else None) for n in PROJECTION_SIZES},
        "elapsed_s": elapsed,
    }


def shuffled(units: Iterable[Unit], seed: int | None) -> list[Unit]:
    out = list(units)
    if seed is not None:
        random.Random(seed).shuffle(out)
    return out


def label_units(
    units: Iterable[Unit],
    variant: Variant,
    cfg: EngineConfig,
    run_dir: Path,
    limit: int | None = None,
    batch_size: int = 64,
    *,
    llm: Any = None,
    sampling: Any = None,
    cache: TeacherCache | None = None,
) -> dict[str, Any]:
    """Label ``units`` (first ``limit``), skipping cached ones. Returns final metrics.

    ``llm``/``sampling``/``cache`` are injectable for tests; defaults build the real ones.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    todo = list(units)[:limit] if limit is not None else list(units)
    cache = cache or TeacherCache.for_model(cfg.model_slug)
    sj = cfg.sampling_json()
    keyed = [
        (
            u,
            build_messages(u, variant),
        )
        for u in todo
    ]
    keys = [
        cache_key(cfg.model, cfg.revision, variant, m[0]["content"], m[1]["content"], sj)
        for _, m in keyed
    ]
    pending = [i for i, k in enumerate(keys) if cache.get(k) is None]
    n_done = len(todo) - len(pending)
    n_valid = sum(1 for k in keys if (r := cache.get(k)) is not None and r.valid)
    gen = prompt = new_units = 0
    t0 = time.time()
    if pending and llm is None:
        llm = build_llm(cfg)
        sampling = build_sampling(cfg)
    metrics = _metrics(n_done, n_valid, 0, 0, 0, 0.0)
    for s in range(0, len(pending), batch_size):
        idx = pending[s : s + batch_size]
        outs = llm.chat(
            [keyed[i][1] for i in idx], sampling, chat_template_kwargs=CHAT_TEMPLATE_KWARGS
        )
        rows = []
        for i, o in zip(idx, outs, strict=True):
            c = o.outputs[0]
            text, ok = validate_output(c.text, keyed[i][0].unit_id)
            npt = len(getattr(o, "prompt_token_ids", None) or [])
            ngt = len(c.token_ids)
            rows.append(CacheRow(keys[i], keyed[i][0].unit_id, variant, text, ok, npt, ngt))
            gen += ngt
            prompt += npt
            n_valid += ok
        cache.put_many(rows)
        n_done += len(idx)
        new_units += len(idx)
        metrics = _metrics(n_done, n_valid, new_units, gen, prompt, time.time() - t0)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(
            f"[{n_done}/{len(todo)}] valid={metrics['valid_rate']:.3f} "
            f"tok/s={metrics['tok_per_s']:.0f} u/h={metrics['units_per_hour']:.0f}",
            flush=True,
        )
    n_obl = n_unres = n_wrong = n_empty = 0
    rows_out: list[dict[str, Any]] = []
    for (u, _), k in zip(keyed, keys, strict=True):
        r = cache.get(k)
        assert r is not None
        extraction: dict[str, Any] | None = None
        n_u = wrong = 0
        if r.valid:
            res = resolve(
                GenExtraction.model_validate_json(r.output), u.unit_id, u.source, u.text, u.heading
            )
            extraction = res.extraction.model_dump(mode="json")
            n_u, wrong = len(res.unresolved), res.wrong_unit
            n_obl += len(res.extraction.obligations) + n_u
            n_unres += n_u
            n_wrong += wrong
            n_empty += not res.extraction.obligations and not n_u
        rows_out.append(
            {
                "unit_id": u.unit_id,
                "variant": variant,
                "valid": r.valid,
                "extraction": extraction,
                "n_unresolved": n_u,
                "wrong_unit": wrong,
            }
        )
    metrics = {
        **metrics,
        "obligations_total": n_obl,
        "unresolved_rate": n_unres / n_obl if n_obl else 0.0,
        "wrong_unit_rate": n_wrong / n_obl if n_obl else 0.0,
        "empty_rate": n_empty / n_valid if n_valid else 0.0,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (run_dir / "labels.jsonl").open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics
