"""CPU tests for regslm.train.grpo (docs/specs/grpo.md)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_train import mk_label, mk_unit

from regslm.rewards import reward
from regslm.schema.obligation import Extraction, Obligation, Threshold
from regslm.train import data as D
from regslm.train.config import TrainConfig
from regslm.train.grpo import (
    BudgetCallback,
    EvalCallback,
    GrpoConfig,
    completion_text,
    default_num_generations,
    hacking_flags,
    make_reward_fn,
    prompt_rows,
    run_grpo,
    sample_prompts,
    select_parent,
    write_run_config,
)


def numeric_label(u: Any) -> D.LabelRow:
    r = mk_label(u)
    assert r.extraction is not None
    o = r.extraction.obligations[0]
    o2 = Obligation(
        **{**o.model_dump(), "threshold": Threshold(value=5, unit="%", comparator=">=")}
    )
    r.extraction = Extraction(unit_id=u.unit_id, obligations=[o2])
    return r


def gen(u: Any, quote: str | None = None) -> str:
    return json.dumps(
        {
            "unit_id": u.unit_id,
            "obligations": [
                {
                    "subject": "Institutions", "modality": "must", "action": "report",
                    "object": "the amount",
                    "citation": {
                        "source": u.source, "unit_id": u.unit_id,
                        "quote": quote or u.text.split(".")[0] + ".",
                    },
                }
            ],
        }
    )  # fmt: skip


def test_reward_fn_alignment_and_logging(tmp_path: Path) -> None:
    us = [mk_unit(i) for i in range(3)]
    rows = {u.unit_id: mk_label(u) for u in us}
    ref_json = [rows[u.unit_id].extraction.model_dump_json() for u in us]  # type: ignore[union-attr]
    unit_json = [u.model_dump_json() for u in us]
    comps: list[Any] = [
        [{"role": "assistant", "content": gen(us[0])}],  # chat format
        "not json",  # plain str
        gen(us[2]),
    ]
    log = tmp_path / "rc.jsonl"
    metrics: dict[str, float] = {}
    fn = make_reward_fn(None, log)
    out = fn(
        prompts=["p"] * 3, completions=comps, unit_json=unit_json, reference_json=ref_json,
        trainer_state=SimpleNamespace(global_step=7), log_metric=metrics.__setitem__,
    )  # fmt: skip
    expect = [
        reward(completion_text(c), u, rows[u.unit_id].extraction).total
        for c, u in zip(comps, us, strict=True)
    ]
    assert out == pytest.approx(expect)
    assert out[1] == 0.0 and out[0] > 0.5 and out[2] > 0.5
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["step"] == 7 and rec["n"] == 3 and "reward/num" in rec
    assert rec["reward/mean_obligations"] == pytest.approx(2 / 3)
    assert "reward/empty_frac" in metrics
    with pytest.raises(ValueError, match="misaligned"):
        fn(prompts=[], completions=comps, unit_json=unit_json[:2], reference_json=ref_json)


def test_hacking_flags() -> None:
    base = {"step": 0, "reward": 0.5, "recall": 0.6, "empty_frac": 0.05}
    ok = {"step": 100, "reward": 0.6, "recall": 0.62, "empty_frac": 0.10}
    assert not hacking_flags([base, ok])["hacked"]
    empty = {**ok, "step": 250, "empty_frac": 0.15}
    r = hacking_flags([empty, base, ok])
    assert r["hacked"] and r["reasons"] == [{"step": 250, "rule": "empty_rate_up_10pp"}]
    rec = {**ok, "recall": 0.54}
    assert hacking_flags([base, rec])["reasons"][0]["rule"] == "recall_down_5pp_reward_up"
    assert not hacking_flags([base, {**rec, "reward": 0.4}])["hacked"]  # reward fell: no flag
    assert not hacking_flags([ok])["hacked"]  # no step-0 baseline


def test_budget_callback_stops() -> None:
    t = [0.0]
    cb = BudgetCallback(100.0, clock=lambda: t[0])
    ctl = SimpleNamespace(should_training_stop=False, should_save=False)
    cb.on_step_end(None, None, ctl)
    assert not ctl.should_training_stop and not cb.stopped
    t[0] = 101.0
    cb.on_step_end(None, None, ctl)
    assert ctl.should_training_stop and ctl.should_save and cb.stopped


def test_eval_callback_steps() -> None:
    calls: list[int] = []
    cb = EvalCallback([0, 2], lambda m, s: calls.append(s) or {"step": s})  # type: ignore[func-returns-value]
    for s in (0, 1, 2):
        st = SimpleNamespace(global_step=s)
        (cb.on_train_begin if s == 0 else cb.on_step_end)(None, st, None, model=object())
    assert calls == [0, 2]


def test_sampler_numeric_share_and_overlap() -> None:
    us = {f"x/U{i}": mk_unit(i) for i in range(100)}
    rows = {
        u.unit_id: numeric_label(u) if i % 4 == 0 else mk_label(u)
        for i, u in enumerate(us.values())
    }
    rows["x/U1"] = mk_label(us["x/U1"], n_unres=1)  # unresolved -> excluded
    sft = {u for i, u in enumerate(us) if i % 2 == 0}  # includes all numeric ones
    ids, rep = sample_prompts(rows, us, sft, 20, 0.2, seed=0)
    assert len(ids) == len(set(ids)) == 20 and "x/U1" not in ids
    assert rep["n_numeric"] == 4 and rep["numeric_share"] == pytest.approx(0.2)
    # numeric units only exist inside the SFT subset: 4 of the 20 overlap, the rest are fresh
    assert rep["n_in_sft_subset"] == 4 and rep["overlap_fraction"] == pytest.approx(0.2)
    assert ids == sample_prompts(rows, us, sft, 20, 0.2, seed=0)[0]  # deterministic
    # pool smaller than n: take everything, overlap counted
    ids2, rep2 = sample_prompts(rows, us, sft, 500, 0.2)
    assert len(ids2) == rep2["n_eligible"] == 99
    assert rep2["overlap_fraction"] == pytest.approx(rep2["n_in_sft_subset"] / 99)


def test_prompt_rows_columns() -> None:
    u = mk_unit(1)
    r = prompt_rows([u.unit_id], {u.unit_id: u}, {u.unit_id: mk_label(u)})[0]
    assert r["prompt"][0]["role"] == "system" and u.text in r["prompt"][1]["content"]
    assert json.loads(r["unit_json"])["unit_id"] == u.unit_id
    assert Extraction.model_validate_json(r["reference_json"]).unit_id == u.unit_id


def _parent(runs: Path, rid: str, n: int, of1: float | None, base: str = "Qwen/Qwen3-0.6B") -> None:
    TrainConfig(run_id=rid, base=base, n_labels=n).save(runs)
    if of1 is not None:
        d = runs / rid
        (d / "eval.json").write_text(
            json.dumps({"sets": {"silver_val": {"metrics": {"of1": of1}}}})
        )


def test_select_parent_and_config(tmp_path: Path) -> None:
    _parent(tmp_path, "a", 2000, 0.5)
    _parent(tmp_path, "b", 500, 0.5)  # tie -> smaller N
    _parent(tmp_path, "c", 100, 0.4)
    _parent(tmp_path, "d", 100, None)  # not evaluated
    _parent(tmp_path, "e", 100, 0.9, base="Qwen/Qwen3-4B")
    assert select_parent("Qwen/Qwen3-0.6B", tmp_path) == "b"
    with pytest.raises(LookupError):
        select_parent("nope", tmp_path)
    g = write_run_config("g1", "e", {"max_steps": 5}, tmp_path)
    assert g.num_generations == 4 == default_num_generations("Qwen/Qwen3-4B") and g.beta == 0.04
    assert GrpoConfig.load("g1", tmp_path).max_steps == 5
    assert TrainConfig.load("g1", tmp_path).base == "Qwen/Qwen3-4B"  # predict-compatible
    assert select_parent("Qwen/Qwen3-4B", tmp_path) == "e"  # GRPO run excluded from SFT parents


def test_run_grpo_oom_retry(tmp_path: Path) -> None:
    _parent(tmp_path, "p", 100, 0.5, base="Qwen/Qwen3-4B")
    g = write_run_config("g", "p", {}, tmp_path)
    seen: list[tuple[str, int]] = []

    def fake(cfg: GrpoConfig, d: Path) -> dict[str, Any]:
        seen.append((cfg.run_id, cfg.num_generations))
        if cfg.num_generations > 2:
            raise RuntimeError("CUDA out of memory")
        return {"status": "ok"}

    m = run_grpo(g, tmp_path, fake)
    assert m["status"] == "does_not_fit" and m["retry_run"] == "g-g2"
    assert seen == [("g", 4), ("g-g2", 2)]
    assert json.loads((tmp_path / "g-g2/metrics.json").read_text())["status"] == "ok"
    assert (tmp_path / "g-g2/grpo.yaml").exists()
