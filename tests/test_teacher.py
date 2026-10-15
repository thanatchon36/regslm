"""Teacher tests with a fake LLM (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def good(uid: str) -> str:
    return json.dumps({"unit_id": uid, "obligations": []})


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
            text = "{not json" if n in self.bad else good("WRONG" if self.wrong_id else uid)
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
