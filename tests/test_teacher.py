"""Teacher tests with a fake LLM (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from regslm.corpus.base import Unit
from regslm.teacher.cache import TeacherCache, cache_key
from regslm.teacher.engine import EngineConfig
from regslm.teacher.prompts import (
    NO_RATIONALE_INSTRUCTION,
    RATIONALE_INSTRUCTION,
    system_prompt,
)
from regslm.teacher.runner import label_units

CFG = EngineConfig(model="fake/model")


def mk(i: int) -> Unit:
    return Unit(
        unit_id=f"s/d/{i}",
        source="s",
        doc_id="d",
        path=[str(i)],
        text=f"Firms shall do thing {i}.",
        language="en",
        version="v",
    )


def good(uid: str, n: int = 0) -> str:
    cites = [
        {"source": "s", "unit_id": uid, "quote": f"Firms shall do thing {n}."},
        {"source": "s", "unit_id": uid, "quote": "never in the text at all"},
        {"source": "s", "unit_id": "other", "quote": f"Firms shall do thing {n}."},
    ]
    obs = [
        {"subject": "Firms", "modality": "must", "action": "do thing", "citation": c} for c in cites
    ]
    return json.dumps({"unit_id": uid, "obligations": obs})


class FakeLLM:
    def __init__(self, bad: set[int] | None = None, wrong_id: bool = False) -> None:
        self.calls = 0
        self.seen: list[str] = []
        self.bad = bad or set()
        self.wrong_id = wrong_id

    def chat(self, msgs: list[Any], sampling: Any, **kw: Any) -> list[Any]:
        assert kw["chat_template_kwargs"] == {"enable_thinking": False}
        out = []
        for m in msgs:
            self.calls += 1
            user = m[1]["content"]
            self.seen.append(user)
            uid = user.split("unit_id: ")[1].split("\n")[0]
            n = int(uid.rsplit("/", 1)[1])
            text = "{not json" if n in self.bad else good("WRONG" if self.wrong_id else uid, n)
            out.append(
                SimpleNamespace(
                    prompt_token_ids=[1] * 10,
                    outputs=[SimpleNamespace(text=text, token_ids=[1] * 5)],
                )
            )
        return out


def run(tmp_path: Path, llm: FakeLLM, units: list[Unit], **kw: Any) -> dict[str, Any]:
    cache = TeacherCache(tmp_path / "c.sqlite")
    return label_units(
        units, "label", CFG, tmp_path / "run", llm=llm, sampling=object(), cache=cache, **kw
    )


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    llm = FakeLLM()
    units = [mk(i) for i in range(5)]
    m = run(tmp_path, llm, units, batch_size=2)
    assert llm.calls == 5 and m["n_done"] == 5 and m["valid_rate"] == 1.0
    run(tmp_path, llm, units)
    assert llm.calls == 5  # all hits
    lines = (tmp_path / "run/labels.jsonl").read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[0])["extraction"]["unit_id"] == "s/d/0"
    row = json.loads(lines[0])
    assert row["n_unresolved"] == 1 and row["wrong_unit"] == 1
    assert len(row["extraction"]["obligations"]) == 2
    assert row["extraction"]["obligations"][0]["citation"]["span"] == {"start": 0, "end": 23}
    assert m["obligations_total"] == 15 and m["unresolved_rate"] == pytest.approx(1 / 3)
    assert m["wrong_unit_rate"] == pytest.approx(1 / 3) and m["empty_rate"] == 0.0


def test_resume_after_partial(tmp_path: Path) -> None:
    llm = FakeLLM()
    run(tmp_path, llm, [mk(i) for i in range(3)])
    assert llm.calls == 3
    run(tmp_path, llm, [mk(i) for i in range(6)], batch_size=4)
    assert llm.calls == 6


def test_limit(tmp_path: Path) -> None:
    llm = FakeLLM()
    run(tmp_path, llm, [mk(i) for i in range(6)], limit=2)
    assert llm.calls == 2


def test_invalid_recorded(tmp_path: Path) -> None:
    llm = FakeLLM(bad={1})
    m = run(tmp_path, llm, [mk(0), mk(1)])
    assert m["n_valid"] == 1 and m["valid_rate"] == 0.5
    rows = [json.loads(x) for x in (tmp_path / "run/labels.jsonl").read_text().splitlines()]
    assert rows[1]["valid"] is False and rows[1]["extraction"] is None
    run(tmp_path, llm, [mk(0), mk(1)])
    assert llm.calls == 2  # invalid not retried


def test_unit_id_forced(tmp_path: Path) -> None:
    run(tmp_path, FakeLLM(wrong_id=True), [mk(0)])
    row = json.loads((tmp_path / "run/labels.jsonl").read_text())
    assert row["extraction"]["unit_id"] == "s/d/0"


def test_metrics_shape(tmp_path: Path) -> None:
    run(tmp_path, FakeLLM(), [mk(0), mk(1)])
    m = json.loads((tmp_path / "run/metrics.json").read_text())
    for k in (
        "n_done",
        "n_valid",
        "valid_rate",
        "gen_tokens",
        "prompt_tokens",
        "tok_per_s",
        "units_per_hour",
        "projected_hours_for_30000",
        "projected_hours_for_50000",
        "elapsed_s",
        "obligations_total",
        "unresolved_rate",
        "wrong_unit_rate",
        "empty_rate",
    ):
        assert k in m
    assert m["gen_tokens"] == 10 and m["prompt_tokens"] == 20


def test_prompt_variants() -> None:
    sys = {v: system_prompt(v) for v in ("label", "rationale", "gold_a", "gold_b")}  # type: ignore[misc]
    assert len(set(sys.values())) == 4
    assert sys["label"].replace(NO_RATIONALE_INSTRUCTION, "") == sys["rationale"].replace(
        RATIONALE_INSTRUCTION, ""
    )


def test_cache_key_sensitivity() -> None:
    a = cache_key("m", None, "label", "s", "u", "{}")
    assert a == cache_key("m", None, "label", "s", "u", "{}")
    assert a != cache_key("m", None, "rationale", "s", "u", "{}")
    assert a != cache_key("m2", None, "label", "s", "u", "{}")
    assert a != cache_key("m", None, "label", "s", "u", '{"t":1}')


class _Tok:
    """Prompt length = 10 tokens, except unit 1 whose prompt exceeds the context."""

    def apply_chat_template(self, msgs: list[Any], **kw: Any) -> list[int]:
        return [1] * (10_000 if "unit_id: s/d/1\n" in msgs[1]["content"] else 10)


class _Params:
    def __init__(self) -> None:
        self.max_tokens = 1024

    def clone(self) -> _Params:
        return _Params()


class FakeLLMWithTokenizer(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.params: list[Any] = []

    def get_tokenizer(self) -> _Tok:
        return _Tok()

    def chat(self, msgs: list[Any], sampling: Any, **kw: Any) -> list[Any]:
        self.params.extend(sampling if isinstance(sampling, list) else [sampling])
        return super().chat(msgs, sampling, **kw)


def test_too_long_units_are_recorded_not_sent(tmp_path: Path) -> None:
    llm = FakeLLMWithTokenizer()
    cache = TeacherCache(tmp_path / "c.sqlite")
    units = [mk(i) for i in range(3)]
    m = label_units(units, "label", CFG, tmp_path / "run", llm=llm, sampling=_Params(), cache=cache)
    assert llm.calls == 2  # unit 1 never sent
    assert m["n_done"] == 3
    assert all(p.max_tokens == min(CFG.max_tokens, CFG.max_model_len - 10) for p in llm.params)
    rows = {
        json.loads(ln)["unit_id"]: json.loads(ln)
        for ln in (tmp_path / "run/labels.jsonl").read_text().splitlines()
    }
    assert rows["s/d/1"]["valid"] is False and rows["s/d/1"]["extraction"] is None
