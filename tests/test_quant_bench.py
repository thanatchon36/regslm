"""CPU tests for regslm.quant / regslm.bench: stub HTTP server, fake tokenizer, no GPU."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from test_train import mk_label, mk_unit

from regslm.bench import aggregate as A
from regslm.bench import needle as N
from regslm.bench import runner as R
from regslm.quant import export as Q
from regslm.train.evaluate import evaluate
from regslm.train.predict import Item

# ------------------------------------------------------------------ stub llama-server

SEEN: list[dict[str, Any]] = []


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:  # silence
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        user = body["messages"][-1]["content"]
        out = json.dumps(
            {
                "choices": [{"message": {"content": f"echo:{user}"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": len(user)},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def stub_url() -> Any:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    SEEN.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_llama_client_chat(stub_url: str) -> None:
    c = R.LlamaClient(stub_url)
    c.wait_ready(timeout=5)
    g = c.chat([{"role": "user", "content": "abc"}])
    assert g.text == "echo:abc" and g.n_prompt == 7 and g.n_gen == 3 and g.latency_s > 0
    req = SEEN[0]
    assert req["temperature"] == 0.0 and req["chat_template_kwargs"] == {"enable_thinking": False}
    assert "grammar" not in req and "response_format" not in req and "json_schema" not in req


def test_llama_client_concurrent_keeps_order(stub_url: str) -> None:
    c = R.LlamaClient(stub_url)
    batch = [[{"role": "user", "content": str(i)}] for i in range(12)]
    gens = c.generate(batch, concurrency=4)
    assert [g.text for g in gens] == [f"echo:{i}" for i in range(12)]


def test_server_cmd_gpu_layers() -> None:
    cpu = R.llama_server_cmd(
        Path("b"), Path("m.gguf"), 1, gpu=False, threads=6, ctx=4096, parallel=2
    )
    gpu = R.llama_server_cmd(
        Path("b"), Path("m.gguf"), 1, gpu=True, threads=6, ctx=4096, parallel=2
    )
    assert cpu[cpu.index("--n-gpu-layers") + 1] == "0"
    assert gpu[gpu.index("--n-gpu-layers") + 1] == "999"
    assert cpu[cpu.index("--threads") + 1] == "6"


# ------------------------------------------------------------------ protocol + predictions


def _items(n: int) -> list[Item]:
    out = []
    for i in range(n):
        u = mk_unit(i)
        ref = mk_label(u).extraction
        assert ref is not None
        out.append(Item(u, ref.model_dump(mode="json")))
    return out


def _oracle(items: list[Item]) -> R.GenerateFn:
    by_text = {R.student_messages(it.unit)[1]["content"]: it for it in items}

    def gen(batch: Any, concurrency: int) -> list[A.Gen]:
        res = []
        for m in batch:
            it = by_text[m[1]["content"]]
            ref = it.reference or {}
            ob = ref["obligations"][0]
            span = ob["citation"]["span"]
            comp = {
                "unit_id": it.unit.unit_id,
                "rationale": None,
                "obligations": [
                    {
                        **{k: v for k, v in ob.items() if k != "citation"},
                        "citation": {
                            "source": it.unit.source,
                            "unit_id": it.unit.unit_id,
                            "quote": it.unit.text[span["start"] : span["end"]],
                        },
                    }
                ],
            }
            res.append(A.Gen(json.dumps(comp), 10, 20, 0.1 * concurrency))
        return res

    return gen


def test_protocol_predictions_feed_evaluate(tmp_path: Path) -> None:
    items = _items(8)
    out = R.run_protocol(_oracle(items), items, 4, warmup=2, n_throughput=6, n_latency=3)
    assert len(out["gens"]) == 6 and out["throughput"]["concurrency"] == 4
    assert out["single_stream"]["n"] == 3
    R.write_predictions(
        tmp_path / "r" / "predictions" / "test.jsonl", "test", out["items"], out["gens"]
    )
    rows = [
        json.loads(ln) for ln in (tmp_path / "r/predictions/test.jsonl").read_text().splitlines()
    ]
    assert {"set", "unit_id", "completion", "valid", "extraction", "n_gen_tokens", "latency_s",
            "reference", "memorisation_targets"} <= set(rows[0])  # fmt: skip
    assert all(r["valid"] for r in rows)
    m = evaluate("r", ["test"], 20, tmp_path)["sets"]["test"]["metrics"]
    assert m["of1"] == pytest.approx(1.0) and m["validity_rate"] == 1.0


# ------------------------------------------------------------------ aggregation


def test_timing_summary() -> None:
    gens = [A.Gen("", 5, 10, x) for x in (1.0, 2.0, 3.0, 4.0, 5.0)]
    s = A.timing_summary(gens, wall_s=2.5)
    assert s["units_per_s"] == 2.0 and s["gen_tok_per_s"] == 20.0
    assert s["latency_s"]["p50"] == 3.0 and s["latency_s"]["p95"] == pytest.approx(4.8)


def test_vram_peak_and_vmhwm(tmp_path: Path) -> None:
    f = tmp_path / "gpu.csv"
    f.write_text(
        "timestamp, memory.used [MiB]\nt0, 1400 MiB\nt1, 1300 MiB\nt2, 9000 MiB\nt3, 5000 MiB\n"
    )
    v = A.vram_peak(f)
    assert v == {"peak_mib": 9000.0, "baseline_mib": 1300.0, "peak_minus_baseline_mib": 7700.0}
    assert A.vram_peak(tmp_path / "none.csv")["peak_mib"] is None
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "status").write_text("Name:\tx\nVmHWM:\t  123456 kB\nVmRSS:\t 1 kB\n")
    assert A.read_vmhwm_kib(42, tmp_path) == 123456 and A.read_vmhwm_kib(43, tmp_path) is None


def test_count_mismatches_and_table(tmp_path: Path) -> None:
    a = [{"unit_id": "1", "completion": "x"}, {"unit_id": "2", "completion": "y"}]
    b = [{"unit_id": "1", "completion": "x"}, {"unit_id": "2", "completion": "z"}]
    assert A.count_mismatches(a, b)["n_mismatch"] == 1
    A.write_result(tmp_path, "vllm", "fp8", {"backend": "vllm", "format": "fp8",
        "throughput": {"units_per_s": 2.0}, "single_stream": {"latency_s": {"p50": 1.0, "p95": 2.0}},
        "eval": {"metrics": {"of1": 0.5}}, "memory": {"vram_peak_minus_baseline_mib": 100.0}})  # fmt: skip
    rows = A.collect(tmp_path)
    assert rows[0]["p95_s"] == 2.0 and rows[0]["of1"] == 0.5
    assert "vllm" in A.to_markdown(rows)


# ------------------------------------------------------------------ needle


class FakeTok:
    """Whitespace tokenizer: one token per word; ids are word indices into a vocab."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids = []
        for w in text.split(" "):
            if w not in self.vocab:
                self.vocab[w] = len(self.inv)
                self.inv.append(w)
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inv[i] for i in ids)


def _doc(n: int) -> list[Any]:
    return [mk_unit(i) for i in range(n)]


@pytest.mark.parametrize("length", [200, 500, 1000])
@pytest.mark.parametrize("depth", [0.1, 0.5, 0.9])
def test_needle_exact_length_and_depth(length: int, depth: int) -> None:
    tok = FakeTok()
    target = mk_unit(999)
    n = N.build_context(_doc(200), target, tok, length, depth)
    assert n.n_tokens == length and len(tok.encode(n.context)) == length
    nb = len(tok.encode(N.unit_block(target)))
    assert n.needle_start_token == round(depth * (length - nb))
    words = n.context.split(" ")
    assert " ".join(words[n.needle_start_token : n.needle_start_token + nb]) == N.unit_block(target)
    assert "unit_id: x/U999" in n.context and "x/U999" in n.messages[1]["content"][-60:]


def test_needle_errors_and_pick_document() -> None:
    tok = FakeTok()
    with pytest.raises(ValueError, match="filler"):
        N.build_context(_doc(3), mk_unit(999), tok, 100_000, 0.5)
    with pytest.raises(ValueError, match="depth"):
        N.build_context(_doc(3), mk_unit(999), tok, 50, 1.5)
    us = _doc(30)
    assert N.pick_document(us, 100, tok) == us


def test_score_needle() -> None:
    u = mk_unit(1)
    ref = mk_label(u).extraction
    assert ref is not None
    ob = ref.obligations[0]
    quote = u.text[: u.text.index(".") + 1]

    def comp(unit_ids: list[str]) -> str:
        obs = [{**{k: v for k, v in ob.model_dump(mode="json").items() if k != "citation"},
                "citation": {"source": "x", "unit_id": uid, "quote": quote}} for uid in unit_ids]  # fmt: skip
        return json.dumps({"unit_id": u.unit_id, "rationale": None, "obligations": obs})

    ref_d = ref.model_dump(mode="json")
    good = N.score_needle(comp([u.unit_id]), u, ref_d)
    assert good["valid"] and good["of1"] == pytest.approx(1.0) and good["n_other_unit"] == 0
    extra = N.score_needle(comp([u.unit_id, "x/U7", "x/U8"]), u, ref_d)
    assert extra["n_other_unit"] == 2
    bad = N.score_needle("not json", u, ref_d)
    assert not bad["valid"] and bad["of1"] == 0.0


# ------------------------------------------------------------------ quant


def test_quant_formats_and_jobs() -> None:
    assert Q.parse_formats("fp8, gguf-q8,gguf-q4km") == ["fp8", "gguf-q8", "gguf-q4km"]
    with pytest.raises(ValueError):
        Q.parse_formats("fp8,int3")
    jobs = Q.plan_jobs(["gguf-q8", "awq", "fp8", "gguf-q4km"])
    assert [j.fmts for j in jobs] == [["awq"], ["fp8"], ["gguf-q8", "gguf-q4km"]]


def test_calibration_never_gold_and_deterministic() -> None:
    units = {mk_unit(i).unit_id: mk_unit(i) for i in range(50)}
    gold = {"x/U3", "x/U4", "x/U5"}
    a = Q.select_calibration_units(units, gold, n=49)
    assert not gold & {u.unit_id for u in a} and len(a) == 47
    a = Q.select_calibration_units(units, gold, n=20)
    b = Q.select_calibration_units(units, gold, n=20)
    assert [u.unit_id for u in a] == [u.unit_id for u in b] and len(a) == 20


def test_gguf_commands() -> None:
    c = Q.convert_cmd(Path("m"), Path("o.gguf"))
    assert c[1].endswith("convert_hf_to_gguf.py") and c[-2:] == ["--outtype", "f16"]
    q = Q.quantize_cmd(Path("f.gguf"), Path("o.gguf"), Q.GGUF_TYPES["gguf-q4km"])
    assert q[-1] == "Q4_K_M" and q[0].endswith("llama-quantize")
    assert Q.GGUF_TYPES["gguf-q8"] == "Q8_0"
