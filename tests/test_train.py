from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from regslm.corpus.base import Unit
from regslm.schema.generation import GenExtraction, resolve
from regslm.schema.obligation import Citation, Extraction, Obligation, Span
from regslm.train import data as D
from regslm.train.config import TrainConfig, total_steps
from regslm.train.prompts import STUDENT_SYSTEM
from regslm.train.sft import is_oom, loss_summary, run_training

TRIG = "Institutions shall report the amount to the authority."
NOTRIG = "'Institution' means an undertaking that takes deposits."


def mk_unit(i: int, trig: bool = True) -> Unit:
    return Unit(
        unit_id=f"x/U{i}", source="x", doc_id="D", path=["a"], heading="H",
        text=(TRIG if trig else NOTRIG) + f" ({i})", language="en", version="v",
    )  # fmt: skip


def mk_label(
    u: Unit, rationale: str | None = None, n_unres: int = 0, valid: bool = True
) -> D.LabelRow:
    end = u.text.index(".") + 1
    ob = Obligation(
        subject="Institutions", modality="must", action="report", object="the amount",
        citation=Citation(source=u.source, unit_id=u.unit_id, span=Span(start=0, end=end)),
    )  # fmt: skip
    ex = Extraction(unit_id=u.unit_id, rationale=rationale, obligations=[ob])
    return D.LabelRow(u.unit_id, valid, ex if valid else None, n_unres)


def fake_counts() -> tuple[D.CountPrompt, D.CountText]:
    return (lambda msgs: sum(len(m["content"].split()) for m in msgs)), (lambda s: len(s.split()))


def build(units: dict[str, Unit], labels: dict[str, D.LabelRow], **kw: Any) -> D.Built:
    cp, ct = fake_counts()
    args: dict[str, Any] = {
        "n_labels": 10, "arm": "label", "seed": 0, "labels": labels, "rationale_labels": None,
        "train_units": units, "count_prompt": cp, "count_text": ct,
    }  # fmt: skip
    return D.build_examples(**{**args, **kw})


@pytest.fixture
def pool() -> tuple[dict[str, Unit], dict[str, D.LabelRow]]:
    trig = [mk_unit(i) for i in range(40)]
    notrig = [mk_unit(100 + i, trig=False) for i in range(20)]
    units = {u.unit_id: u for u in trig + notrig}
    return units, {u.unit_id: mk_label(u) for u in trig}


def test_quote_reconstruction_roundtrip() -> None:
    u = mk_unit(1)
    row = mk_label(u)
    assert row.extraction is not None
    g = D.target_from_label(row.extraction, u.text)
    assert g.obligations[0].citation.quote == TRIG
    assert g.rationale is None
    res = resolve(GenExtraction.model_validate_json(D.target_json(g)), u.unit_id, u.source, u.text)
    assert res.extraction.obligations[0].citation.span == Span(start=0, end=len(TRIG))
    assert not res.unresolved


def test_target_is_compact_schema_order() -> None:
    u = mk_unit(1)
    row = mk_label(u, rationale="because shall")
    assert row.extraction is not None
    t = D.target_json(D.target_from_label(row.extraction, u.text, "because shall"))
    assert t.startswith('{"unit_id":"x/U1","rationale":"because shall","obligations":[{"subject"')
    assert "\n" not in t and ": " not in t
    assert D.empty_target("a") == '{"unit_id":"a","rationale":null,"obligations":[]}'


def test_nested_subsets(pool: Any) -> None:
    units, labels = pool
    ids = {n: {e.unit_id for e in build(units, labels, n_labels=n).examples if e.origin == "teacher"}
           for n in (5, 10, 20, 40)}  # fmt: skip
    assert ids[5] < ids[10] < ids[20] < ids[40]
    assert [len(v) for v in ids.values()] == [5, 10, 20, 40]
    # independent of the training seed
    other = {
        e.unit_id
        for e in build(units, labels, n_labels=10, seed=7).examples
        if e.origin == "teacher"
    }
    assert other == ids[10]


def test_rule_empty_ratio_and_origin(pool: Any) -> None:
    units, labels = pool  # 40 triggered, 20 untriggered -> ratio 0.5
    b = build(units, labels, n_labels=20)
    assert b.report["rule_empty_ratio"] == pytest.approx(0.5)
    assert b.report["n_teacher"] == 20 and b.report["n_rule_empty"] == 10
    empties = [e for e in b.examples if e.origin == "rule_empty"]
    assert all(json.loads(e.target)["obligations"] == [] for e in empties)
    assert all(e.unit_id.startswith("x/U1") for e in empties)
    assert build(units, labels, n_labels=20, rule_empty=False).report["n_rule_empty"] == 0


def test_drops_invalid_unresolved_nontrain(pool: Any) -> None:
    units, labels = pool
    ids = sorted(labels)
    labels[ids[0]] = mk_label(units[ids[0]], valid=False)
    labels[ids[1]] = mk_label(units[ids[1]], n_unres=1)
    labels["x/outside"] = mk_label(mk_unit(999))
    b = build(units, labels, n_labels=100)
    assert b.report["dropped"] == {"invalid_label": 1, "unresolved_obligations": 1, "not_train": 1}
    got = {e.unit_id for e in b.examples}
    assert ids[0] not in got and ids[1] not in got and "x/outside" not in got


def test_too_long_dropped_and_counted(pool: Any) -> None:
    units, labels = pool
    b = build(units, labels, n_labels=10, max_seq_len=5, rule_empty=False)
    assert b.examples == [] and b.report["dropped_too_long"] == 10


def test_placebo_permutes_targets(pool: Any) -> None:
    units, labels = pool
    for i, (k, r) in enumerate(sorted(labels.items())):  # distinct payloads
        assert r.extraction is not None
        ob = r.extraction.obligations[0].model_copy(update={"action": f"act{i}"})
        labels[k] = D.LabelRow(k, True, r.extraction.model_copy(update={"obligations": [ob]}))
    real = build(units, labels, n_labels=30)
    plc = build(units, labels, n_labels=30, arm="placebo", seed=3)

    def payload(e: D.Example) -> str:
        d = json.loads(e.target)
        d.pop("unit_id")
        return json.dumps(d, sort_keys=True)

    assert sorted(map(payload, real.examples)) == sorted(map(payload, plc.examples))
    assert [payload(e) for e in real.examples] != [payload(e) for e in plc.examples]
    assert [e.unit_id for e in real.examples] == [e.unit_id for e in plc.examples]
    assert all(json.loads(e.target)["unit_id"] == e.unit_id for e in plc.examples)
    again = build(units, labels, n_labels=30, arm="placebo", seed=3)
    assert [e.target for e in again.examples] == [e.target for e in plc.examples]


def test_rationale_arm_and_token_budget_matching(pool: Any) -> None:
    units, labels = pool
    rat = {k: mk_label(units[k], rationale="trigger word shall appears so this is a duty " * 2)
           for k in labels}  # fmt: skip
    r = build(units, labels, rationale_labels=rat, arm="rationale", n_labels=10)
    assert all(json.loads(e.target)["rationale"] for e in r.examples if e.origin == "teacher")
    plain = build(units, labels, n_labels=10)
    budget = r.report["target_tokens"]
    assert plain.report["target_tokens"] < budget
    m = build(units, labels, rationale_labels=rat, n_labels=10, match_rationale_tokens=True)
    assert m.report["token_budget"] == budget
    assert m.report["target_tokens"] >= budget
    assert m.report["n_labels_selected"] > 10
    # minimal: one fewer label would fall short of the budget
    fewer = build(units, labels, n_labels=m.report["n_labels_selected"] - 1)
    assert fewer.report["target_tokens"] < budget
    # nested: matched set contains the plain N=10 set
    ids = lambda b: {e.unit_id for e in b.examples if e.origin == "teacher"}  # noqa: E731
    assert ids(plain) < ids(m)


def test_rationale_arm_drops_missing_rationale(pool: Any) -> None:
    units, labels = pool
    rat = {k: mk_label(units[k], rationale=None) for k in labels}
    b = build(units, labels, rationale_labels=rat, arm="rationale", n_labels=5)
    assert b.report["dropped"]["no_rationale"] == 40 and b.examples == []


def test_gold_exclusion(tmp_path: Path) -> None:
    gold = tmp_path / "sample.jsonl"
    gold.write_text(json.dumps({"unit_id": "x/G1"}) + "\n")
    D.check_no_gold(["x/A"], gold)
    with pytest.raises(ValueError, match="gold"):
        D.check_no_gold(["x/A", "x/G1"], gold)


def test_config_hash_and_steps() -> None:
    a = TrainConfig(run_id="a")
    assert a.hash() == TrainConfig(run_id="b").hash()
    assert a.hash() != TrainConfig(run_id="a", lr=1e-4).hash()
    assert a.effective_lr == 2e-4 and TrainConfig(run_id="a", method="full").effective_lr == 2e-5
    assert TrainConfig(run_id="a", method="full").effective_optim == "adamw_8bit"
    assert total_steps(500, a) == 300  # 2 epochs = 64 steps < min_steps
    assert total_steps(20000, a) == 2500
    assert total_steps(500, TrainConfig(run_id="a", max_steps=50)) == 50


def test_config_roundtrip(tmp_path: Path) -> None:
    c = TrainConfig(run_id="r1", n_labels=2000, arm="placebo")
    c.save(tmp_path)
    assert TrainConfig.load("r1", tmp_path) == c


def test_oom_is_a_result(tmp_path: Path) -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    def boom(cfg: TrainConfig, run_dir: Path) -> dict[str, Any]:
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    m = run_training(TrainConfig(run_id="oom", method="full"), tmp_path, trainer=boom)
    assert m["status"] == "does_not_fit"
    assert json.loads((tmp_path / "oom" / "metrics.json").read_text())["status"] == "does_not_fit"
    assert is_oom(OutOfMemoryError("x"))


def test_other_errors_propagate(tmp_path: Path) -> None:
    def bad(cfg: TrainConfig, run_dir: Path) -> dict[str, Any]:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        run_training(TrainConfig(run_id="e"), tmp_path, trainer=bad)


def test_loss_summary() -> None:
    h = [{"step": i, "loss": 3.0 - i * 0.1} for i in range(1, 21)] + [{"step": 20, "eval": 1}]
    s = loss_summary(h)
    assert s["n_logged"] == 20 and s["first_loss"] == pytest.approx(2.9)
    assert s["final_loss"] < s["first_loss"] and loss_summary([]) == {"n_logged": 0}


class FakeTok:
    """Minimal chat-template tokenizer: words are tokens."""

    def apply_chat_template(self, msgs: list[dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = False, **kw: Any) -> str:  # fmt: skip
        s = "".join(f"<s>{m['role']} {m['content']}<e>\n" for m in msgs)
        return s + ("<s>assistant " if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [hash(w) % 1000 for w in text.split()]}


def test_completion_suffix_and_encode() -> None:
    t = FakeTok()
    assert D.completion_suffix(t) == "<e>"
    ex = D.Example("u", "teacher", [{"role": "user", "content": "a b"}], "a b")
    enc = D.encode(t, ex, "<e>")
    assert len(enc["input_ids"]) == len(enc["completion_mask"])
    assert enc["completion_mask"][0] == 0 and enc["completion_mask"][-1] == 1
    assert sum(enc["completion_mask"]) == 2  # two target words (turn end glued)


def test_build_from_config_fixture_mode() -> None:
    cfg = TrainConfig(
        run_id="fx", labels=["tests/fixtures/quick_labels.jsonl"], splits=None, eval_n=50,
        n_labels=1000, rule_empty=False,
    )  # fmt: skip
    built, held = D.build_from_config(cfg, FakeTok())
    train_ids = {e.unit_id for e in built.examples}
    assert len(held) == 50 and not ({r.unit_id for r in held} & train_ids)
    assert 100 < len(train_ids) <= 149
    assert built.report["origin_column"] == {"teacher": len(built.examples)}


def test_student_prompt_is_short() -> None:
    transformers = pytest.importorskip("transformers")
    AutoTokenizer = transformers.AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-AWQ", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not cached")
    assert len(tok(STUDENT_SYSTEM)["input_ids"]) <= 120
    assert D.completion_suffix(tok) == "<|im_end|>"
    u = mk_unit(1)
    cp, _ = D.tokenizer_counters(tok, "<|im_end|>")
    from regslm.train.prompts import student_messages

    assert cp(student_messages(u)) > 0
    assert "<think>" in D.render_prompt(tok, student_messages(u))  # empty think block (no thinking)


def test_zero_shot_prompt_is_teacher_prompt_plus_schema() -> None:
    from regslm.corpus.base import Unit
    from regslm.teacher.prompts import system_prompt
    from regslm.train.prompts import zero_shot_messages

    u = Unit(unit_id="s/d/1", source="s", doc_id="d", path=["1"], text="Firms shall act.", language="en", version="v")
    msgs = zero_shot_messages(u)
    assert msgs[0]["content"].startswith(system_prompt("label"))
    assert '"quote"' in msgs[0]["content"] and "unit_id: s/d/1" in msgs[1]["content"]
