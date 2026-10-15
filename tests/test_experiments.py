"""Experiment orchestration (docs/specs/experiments.md): grid, budget, ledger, runner."""

from __future__ import annotations

import builtins
import itertools
import json
import pathlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_train import build, mk_label, mk_unit  # type: ignore[import-not-found]

from experiments import budget, ledger, run, throughput
from experiments import grid as G
from regslm.train import data as D
from regslm.train.config import TrainConfig


@pytest.fixture(scope="module")
def grid() -> dict[str, Any]:
    return G.load_grid()


def legacy_projection(g: dict[str, Any]) -> dict[str, tuple[int, float]]:
    """The pre-refactor experiments/budget.py arithmetic, kept verbatim as an oracle."""
    inf = g["assumed_infer_units_h"]
    tok_s = g["assumed_train_tok_s"]
    tpe = g["tokens_per_example"]
    sets_all = g["eval_sets"]
    out = {}

    def eval_h(base: str, sets: list[str]) -> float:
        return sum(G.EVAL_UNITS[s] for s in sets) / inf[base]

    def train_h(base: str, method: str, n: int, arm: str) -> float:
        n_eff = n * 1.46
        epochs = 1 if n >= 10000 else 3
        tokens = max(epochs * n_eff, 300 * 16) * tpe * (1.6 if arm == "rationale" else 1.0)
        return tokens / tok_s[base] / 3600 * {"full": 1.8, "qlora": 1.4}.get(method, 1.0)

    for b in sorted(g["blocks"], key=lambda b: b["priority"]):
        bh, runs = 0.0, 0
        for c in b["cells"]:
            sets = list(c.get("eval_sets", sets_all)) + list(b.get("extra_eval_sets", []))
            for base, method in itertools.product(
                G.as_list(c["base"]), G.as_list(c.get("method", "lora"))
            ):
                if method == "zero_shot":
                    bh += eval_h(base, sets)
                    runs += 1
                    continue
                if b["name"] == "quantization":
                    nf = len(c["formats"])
                    bh += 0.3 + nf * 2129 / inf[base] * 1.5
                    if base in c.get("cpu_bases", []):
                        bh += 300 / (inf[base] / 15)
                    runs += nf
                    continue
                seeds = g["seeds"] if c.get("seeds") == "all" else c.get("seeds", g["seeds"])
                if method == "grpo":
                    bh += len(seeds) * (c.get("max_wall_h", 2.0) + eval_h(base, sets))
                    runs += len(seeds)
                    continue
                if method == "lora" and base == "qwen3-8b":
                    continue
                for n in G.as_list(c.get("n", 10000)):
                    bh += len(seeds) * (
                        train_h(base, method, n, c.get("arm", "label")) + eval_h(base, sets)
                    )
                    runs += len(seeds)
        out[b["name"]] = (runs, bh)
    return out


def test_expansion_matches_legacy_budget(grid: dict[str, Any]) -> None:
    cells = G.expand(grid)
    got = {n: (r, h) for _, n, r, h in G.block_summary(cells)}
    want = legacy_projection(grid)
    assert got.keys() == want.keys()
    for blk, (runs, h) in want.items():
        assert got[blk][1] == pytest.approx(h), blk
        if blk != "quantization":  # quantization: one run per valid (format, backend) pair
            assert got[blk][0] == runs, blk
    assert sum(h for *_, h in G.block_summary(cells)) == pytest.approx(
        sum(h for _, h in want.values())
    )


def test_run_ids_unique_and_readable(grid: dict[str, Any]) -> None:
    cells = G.expand(grid)
    ids = [c.run_id for c in cells]
    assert len(ids) == len(set(ids))
    by = {c.run_id: c for c in cells}
    assert "zs-qwen3-14b-awq" in by
    assert "grpo-qwen3-4b-s1" in by
    assert "q-qwen3-0.6b-gguf-q4km-llamacpp-cpu" in by
    assert "sft_ladder_10k-qwen3-8b-qlora-n10000-label-en_th-s1" in by
    assert "thai-qwen3-1.7b-lora-n10000-label-en-s0" in by
    assert by["zs-qwen3-14b-awq"].eval_sets == [
        "test", "test_perturbed", "mr_pool_s0", "heldout", "gold", "mr_pool_s1"
    ]
    assert all(c.hf and c.revision for c in cells if c.base != "sealion3-8b")


def test_missing_revision_refused(grid: dict[str, Any]) -> None:
    revs = {
        k: v for k, v in G.load_revisions().items() if "Qwen3-0.6B" not in k and "SEA-LION" not in k
    }
    with pytest.raises(KeyError, match=r"Qwen/Qwen3-0\.6B"):
        G.expand(grid, revisions=revs, strict=True)
    unpinned = [c for c in G.expand(grid, revisions=revs) if not c.pinned]
    assert {c.base for c in unpinned} == {"qwen3-0.6b", "sealion3-8b"}


def test_cuts_follow_registered_order(grid: dict[str, Any]) -> None:
    base = G.expand(grid)
    cut = G.expand(grid, cuts={"drop_block8", "thai_1seed"})
    assert len(base) - len(cut) == 1 + 3
    assert all(c.priority != 8 for c in cut)
    assert next(k for k, _ in G.CUT_ORDER) == "drop_block8"


def test_nested_subsets_identical_across_bases() -> None:
    units = {u.unit_id: u for u in (mk_unit(i) for i in range(40))}
    labels = {uid: mk_label(u) for uid, u in units.items()}

    def ids(n: int, tok: int) -> set[str]:
        b = build(units, labels, n_labels=n, count_text=lambda s, t=tok: len(s) * t)
        return {e.unit_id for e in b.examples}

    for n in (5, 10, 20):
        assert ids(n, 1) == ids(n, 3)  # tokenizer (base) independent
    assert ids(5, 1) < ids(10, 2) < ids(20, 3)
    assert D.hash_order(units, 0) == D.hash_order(reversed(list(units)), 0)


# ------------------------------------------------------------------------------------ ledger


def _exit(runs: Path, rel: str, wall: float) -> None:
    p = runs / rel / "exit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"run_id": rel, "exit_code": 0, "wall_s": wall}))


def test_ledger_sums(tmp_path: Path, grid: dict[str, Any]) -> None:
    r = tmp_path / "runs"
    _exit(r, "silver-train-label", 3600)
    _exit(r, "gold-pass-a", 1800)
    _exit(r, "teacher-smoke", 1800)
    _exit(r, "quick", 100)
    _exit(r, "zs-qwen3-0.6b/predict", 1800)
    _exit(r, "zs-qwen3-0.6b/evaluate", 1800)
    _exit(r, "sft_ladder_10k-qwen3-1.7b-lora-n10000-label-en_th-s0", 3600)
    _exit(r, "sft_ladder_10k-qwen3-1.7b-lora-n10000-label-en_th-s0/predict", 3600)
    _exit(r, "grpo-qwen3-4b-s0", 7200)
    _exit(r, "q-qwen3-8b-fp8-vllm", 3600)
    out = ledger.write(r, tmp_path / "gpu_ledger.json", {b["name"] for b in grid["blocks"]})
    assert out["teacher_h"] == pytest.approx(2.0)
    assert out["experiment_h"] == pytest.approx(1 + 2 + 2 + 1)
    assert out["by_block"] == pytest.approx(
        {"zero_shot": 1.0, "sft_ladder_10k": 2.0, "grpo": 2.0, "quantization": 1.0}
    )
    assert out["dev_h"] == pytest.approx(100 / 3600)
    assert json.loads((tmp_path / "gpu_ledger.json").read_text()) == out


# ------------------------------------------------------------------------------------ budget


def _no_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    banned = ("eval.json", "metrics.json")
    real_open = builtins.open
    real_read = pathlib.Path.read_text
    real_popen = pathlib.Path.open

    def guard(p: object) -> None:
        if str(p).endswith(banned):
            raise AssertionError(f"budget code opened {p}")

    def fake_open(f: Any, *a: Any, **k: Any) -> Any:
        guard(f)
        return real_open(f, *a, **k)

    def fake_read(self: Path, *a: Any, **k: Any) -> str:
        guard(self)
        return real_read(self, *a, **k)

    def fake_popen(self: Path, *a: Any, **k: Any) -> Any:
        guard(self)
        return real_popen(self, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(pathlib.Path, "read_text", fake_read)
    monkeypatch.setattr(pathlib.Path, "open", fake_popen)


def test_budget_never_reads_metrics(
    tmp_path: Path, grid: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    r = tmp_path / "runs"
    for name in ("eval.json", "metrics.json"):
        (r / "x").mkdir(parents=True, exist_ok=True)
        (r / "x" / name).write_text("{}")
    _exit(r, "zs-qwen3-0.6b", 3600)
    _no_metrics(monkeypatch)
    with pytest.raises(AssertionError):  # the guard itself works
        (r / "x" / "eval.json").read_text()
    led = ledger.compute(r)
    d = budget.decide(grid, 7, led["experiment_h"], lambda c: run.cell_done(c, r))
    assert not d.ok and d.cuts[:2] == ["drop_block8", "thai_1seed"]
    assert run.main(["--dry", "--runs", str(r), "--queues", str(tmp_path / "q")]) == 0
    assert not (tmp_path / "q").exists()


def test_budget_decision_accounts_for_logged_and_done(grid: dict[str, Any]) -> None:
    none_done = lambda c: False  # noqa: E731
    # D-041 eval sets: mr_pool_s1 only in zero_shot / sft_ladder_10k / grpo
    assert budget.decide(grid, 6, 0.0, none_done).ok  # 97.6
    assert not budget.decide(grid, 6, 3.0, none_done).ok  # 100.6
    assert not budget.decide(grid, 7, 0.0, none_done).ok  # 102.9 -> pre-registered cuts apply
    all_done = lambda c: True  # noqa: E731
    assert budget.decide(grid, 8, 99.0, all_done).ok
    assert not budget.decide(grid, 8, 100.5, all_done).ok  # the cap is not soft
    d = budget.decide(grid, 7, 0.0, none_done)
    assert d.projected_h == pytest.approx(102.9, abs=0.1)
    assert d.cuts[:2] == ["drop_block8", "thai_1seed"] and d.sufficient


# ------------------------------------------------------------------------------------ runner


def _grid_file(tmp_path: Path, grid: dict[str, Any], **over: Any) -> Path:
    p = tmp_path / "grid.yaml"
    p.write_text(yaml.safe_dump({**grid, **over}))
    return p


def _args(tmp_path: Path, grid_file: Path | None = None, *extra: str) -> list[str]:
    a = ["--runs", str(tmp_path / "runs"), "--queues", str(tmp_path / "q")]
    return [*a, *(["--grid", str(grid_file)] if grid_file else []), *extra]


def test_emit_zero_shot_and_skip_done(tmp_path: Path) -> None:
    done = tmp_path / "runs" / "zs-qwen3-0.6b" / "evaluate" / "DONE"
    done.parent.mkdir(parents=True)
    done.touch()
    assert run.main(_args(tmp_path, None, "--block", "zero_shot")) == 0
    lines = (tmp_path / "q" / "zero_shot.txt").read_text().splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    assert len(body) == 8  # 4 remaining cells x (predict, evaluate)
    assert not any("zs-qwen3-0.6b " in ln for ln in body)
    assert "predict zs-qwen3-14b-awq --sets test,test_perturbed,mr_pool_s0,heldout,gold,mr_pool_s1" in body
    zs = yaml.safe_load((tmp_path / "runs/zs-qwen3-14b-awq/zero_shot.yaml").read_text())
    assert zs["model"] == "Qwen/Qwen3-14B-AWQ" and zs["prompt"] == "teacher_v5_with_schema"
    assert "mr_pool_s0" in "".join(body)  # supported by predict
    run.main(_args(tmp_path, None, "--block", "zero_shot"))  # idempotent
    assert (tmp_path / "q" / "zero_shot.txt").read_text().splitlines() == lines


def test_emit_training_cells(tmp_path: Path) -> None:
    assert run.main(_args(tmp_path, None, "--block", "distillation")) == 0
    rid = "distillation-qwen3-0.6b-lora-n10000-label_token_matched-en_th-s1"
    cfg = TrainConfig.load(rid, tmp_path / "runs")
    assert cfg.arm == "label" and cfg.match_rationale_tokens and cfg.rationale_labels
    assert cfg.epochs == 1 and cfg.seed == 1 and cfg.base == "Qwen/Qwen3-0.6B"
    assert cfg.revision == G.load_revisions()["Qwen/Qwen3-0.6B"]
    q = (tmp_path / "q" / "distillation.txt").read_text().splitlines()
    i = q.index(f"train {rid}")
    assert q[i + 1].startswith(f"predict {rid} --sets test,") and q[i + 2].startswith("evaluate ")
    assert len([ln for ln in q if ln.startswith("train ")]) == 16


def test_thai_langs_and_unpinned_refusal(tmp_path: Path, grid: dict[str, Any]) -> None:
    gf = _grid_file(tmp_path, grid, budget_gpu_hours=500)
    run.main(_args(tmp_path, gf, "--block", "thai"))
    cfg = TrainConfig.load("thai-qwen3-1.7b-lora-n10000-label-en-s0", tmp_path / "runs")
    assert cfg.langs == ["en"]
    cfg2 = TrainConfig.load("thai-typhoon2-1b-lora-n10000-label-en_th-s0", tmp_path / "runs")
    assert cfg2.langs is None
    assert run.main(_args(tmp_path, gf, "--block", "thai_optional", "--dry")) == 0
    # SEA-LION is pinned in results/base_revisions.json; the unpinned refusal path is covered
    # by test_missing_revision_refused.


def test_over_budget_stops_without_emitting(
    tmp_path: Path, grid: dict[str, Any], capsys: Any
) -> None:
    gf = _grid_file(tmp_path, grid, budget_gpu_hours=10)
    assert run.main(_args(tmp_path, gf, "--block", "sft_ladder_10k")) == run.EXIT_OVER_BUDGET
    out = capsys.readouterr().out
    assert "OVER BUDGET" in out and "(1) drop block 8" in out and "not emitting" in out
    assert not (tmp_path / "q").exists() and not (tmp_path / "runs").exists()


def test_grpo_deferred_then_emitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "resolve_parent", lambda c, runs=run.RUNS: None)
    run.main(_args(tmp_path, None, "--block", "grpo"))
    body = (tmp_path / "q" / "grpo.txt").read_text().splitlines()
    assert all(ln.startswith("#") for ln in body)
    TrainConfig(run_id="parent", base="Qwen/Qwen3-0.6B").save(tmp_path / "runs")
    monkeypatch.setattr(run, "resolve_parent", lambda c, runs=run.RUNS: "parent")
    run.main(_args(tmp_path, None, "--block", "grpo"))
    body = (tmp_path / "q" / "grpo.txt").read_text().splitlines()
    assert "grpo grpo-qwen3-0.6b-s0" in body
    g = yaml.safe_load((tmp_path / "runs/grpo-qwen3-0.6b-s0/grpo.yaml").read_text())
    assert g["parent"] == "parent" and g["max_wall_s"] == 1.5 * 3600


def test_quant_stages(grid: dict[str, Any]) -> None:
    c = next(c for c in G.expand(grid) if c.run_id == "q-qwen3-1.7b-awq-vllm")
    st = run.stages(c, Path("r"), parent="p")
    assert [s.line for s in st] == ["quantize p", "bench-cell q-qwen3-1.7b-awq-vllm p vllm awq"]
    assert st[0].marker == Path("r/p-quantize/DONE")


def test_throughput_reads_only_throughput_fields(tmp_path: Path, grid: dict[str, Any]) -> None:
    r = tmp_path / "runs"
    d = r / "a"
    d.mkdir(parents=True)
    TrainConfig(run_id="a", base="Qwen/Qwen3-1.7B", n_labels=10000).save(r)
    (d / "DONE").touch()
    (d / "metrics.json").write_text(json.dumps({"tokens_seen": 7000 * 3600, "train_wall_s": 3600}))
    (d / "predict_meta.json").write_text(
        json.dumps({"sets": {"test": {"n": 14000, "wall_s": 3600}}})
    )
    m = throughput.write(grid, r, tmp_path / "t.json")
    assert m["train_tok_s"]["qwen3-1.7b"] == pytest.approx(7000)
    assert m["infer_units_h"]["qwen3-1.7b"] == pytest.approx(14000)
    # measured values feed the projection
    slow = G.expand(grid, {"train_tok_s": {"qwen3-1.7b": 3500}})
    base = G.expand(grid)
    assert sum(c.hours for c in slow) > sum(c.hours for c in base)
