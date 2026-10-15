"""Benchmark backends (docs/specs/quant_bench.md): vLLM and llama.cpp (``llama-server``).

Every backend is a ``generate(messages_list, concurrency) -> list[Gen]`` callable; the protocol
(warm-up, offline throughput, single-stream latency) and the predictions file are shared, so
all formats are measured identically. Heavy imports (vLLM) are lazy.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from regslm.bench.aggregate import Gen, read_vmhwm_kib, timing_summary, vram_peak, write_result
from regslm.bench.needle import DEPTHS, LENGTHS, build_context, score_needle
from regslm.train.config import RUNS_DIR, TrainConfig
from regslm.train.predict import MAX_NEW_TOKENS, Item, decode_row, load_set, model_dir
from regslm.train.prompts import student_messages

Messages = list[dict[str, str]]
GenerateFn = Callable[[Sequence[Messages], int], list[Gen]]

WARMUP_N, THROUGHPUT_N, LATENCY_N = 20, 500, 100
THIRD_PARTY = Path("third_party")
LLAMA_CPU_BIN = THIRD_PARTY / "llama.cpp" / "build" / "bin"
LLAMA_CUDA_BIN = THIRD_PARTY / "cuda-b11630" / "llama-b11630"
FORMATS = ("bf16", "fp8-online", "fp8", "awq", "gptq", "gguf-q8", "gguf-q4km")
BACKENDS = ("vllm", "llamacpp-cuda", "llamacpp-cpu")
GGUF_FILES = {"gguf-q8": "Q8_0", "gguf-q4km": "Q4_K_M"}


# ---------------------------------------------------------------- llama.cpp HTTP client


class LlamaClient:
    """OpenAI-compatible chat client for ``llama-server``: temperature 0, no grammar."""

    def __init__(self, base_url: str, timeout: float = 600.0, seed: int = 0) -> None:
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.http = httpx.Client(timeout=timeout)

    def chat(self, messages: Messages, max_tokens: int = MAX_NEW_TOKENS) -> Gen:
        body = {
            "model": "model",
            "messages": messages,
            "temperature": 0.0,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        r = self.http.post(f"{self.base_url}/v1/chat/completions", json=body)
        lat = time.perf_counter() - t0
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage") or {}
        text = d["choices"][0]["message"].get("content") or ""
        return Gen(
            text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), lat
        )

    def generate(self, batch: Sequence[Messages], concurrency: int) -> list[Gen]:
        if concurrency <= 1:
            return [self.chat(m) for m in batch]
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(self.chat, batch))

    def wait_ready(self, timeout: float = 600.0, alive: Callable[[], bool] = lambda: True) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if not alive():
                raise RuntimeError("llama-server exited before becoming ready")
            try:
                if self.http.get(f"{self.base_url}/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError("llama-server not ready")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def llama_server_cmd(
    binary: Path, gguf: Path, port: int, *, gpu: bool, threads: int, ctx: int, parallel: int
) -> list[str]:
    """``--n-gpu-layers 999`` for CUDA, ``0`` for CPU; ``ctx`` is the total across slots."""
    return [
        str(binary / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1",
        "--port", str(port), "--n-gpu-layers", "999" if gpu else "0",
        "--threads", str(threads), "--ctx-size", str(ctx), "--parallel", str(parallel),
        "--no-webui", "--jinja", "--reasoning", "off",
    ]  # fmt: skip


class ServerHandle:
    def __init__(self, proc: subprocess.Popen[bytes], client: LlamaClient) -> None:
        self.proc, self.client = proc, client
        self.peak_rss_kib: int | None = None

    def sample_rss(self) -> None:
        v = read_vmhwm_kib(self.proc.pid)
        if v is not None:
            self.peak_rss_kib = max(self.peak_rss_kib or 0, v)


@contextmanager
def llama_server(
    gguf: Path, *, gpu: bool, threads: int, ctx: int, parallel: int, log: Path | None = None
) -> Any:
    binary = LLAMA_CUDA_BIN if gpu else LLAMA_CPU_BIN
    port = free_port()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{binary.resolve()}:{env.get('LD_LIBRARY_PATH', '')}"
    out = log.open("wb") if log else subprocess.DEVNULL
    proc = subprocess.Popen(
        llama_server_cmd(
            binary.resolve(), gguf, port, gpu=gpu, threads=threads, ctx=ctx, parallel=parallel
        ),
        stdout=out,
        stderr=subprocess.STDOUT,
        env=env,
    )
    h = ServerHandle(proc, LlamaClient(f"http://127.0.0.1:{port}"))
    try:
        h.client.wait_ready(alive=lambda: proc.poll() is None)
        yield h
    finally:
        h.sample_rss()
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------- protocol


def write_predictions(path: Path, name: str, items: Sequence[Item], gens: Sequence[Gen]) -> None:
    """Same row format as ``train/predict.py`` (+ n_prompt_tokens)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it, g in zip(items, gens, strict=True):
            row = {
                "set": name,
                "unit_id": it.unit.unit_id,
                "completion": g.text,
                **decode_row(g.text, it.unit),
                "n_gen_tokens": g.n_gen,
                "n_prompt_tokens": g.n_prompt,
                "latency_s": g.latency_s,
                "reference": it.reference,
                "memorisation_targets": it.memorisation_targets,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_protocol(
    generate: GenerateFn,
    items: Sequence[Item],
    concurrency: int,
    *,
    warmup: int = WARMUP_N,
    n_throughput: int = THROUGHPUT_N,
    n_latency: int = LATENCY_N,
) -> dict[str, Any]:
    """Warm-up, (a) offline throughput at ``concurrency``, (b) single-stream latency."""
    msgs = [student_messages(it.unit) for it in items]
    generate(msgs[:warmup], concurrency)
    thr_items, thr_msgs = items[:n_throughput], msgs[:n_throughput]
    t0 = time.perf_counter()
    thr = generate(thr_msgs, concurrency)
    wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    single = generate(msgs[:n_latency], 1)
    wall1 = time.perf_counter() - t1
    return {
        "items": thr_items,
        "gens": thr,
        "throughput": {**timing_summary(thr, wall), "concurrency": concurrency},
        "single_stream": timing_summary(single, wall1),
        "single_gens": single,
    }


def gguf_path(cfg: TrainConfig, fmt: str, runs: Path = RUNS_DIR) -> Path:
    d = cfg.run_dir(runs) / "quant" / "gguf"
    hits = sorted(d.glob(f"*{GGUF_FILES[fmt]}*.gguf"))
    if not hits:
        raise FileNotFoundError(f"no {GGUF_FILES[fmt]} GGUF in {d}; run `regslm quantize`")
    return hits[0]


def model_path(cfg: TrainConfig, fmt: str, runs: Path = RUNS_DIR) -> Path:
    if fmt in ("bf16", "fp8-online"):
        return model_dir(cfg, runs)
    d = cfg.run_dir(runs) / "quant" / fmt
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"no {fmt} export at {d}; run `regslm quantize`")
    return d


def dir_size_gb(p: Path) -> float:
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e9


# ---------------------------------------------------------------- backends


def vllm_generate_fn(
    model: Path, fmt: str, max_len: int, spec_draft: Path | None = None, spec_k: int = 3,
    gpu_memory_utilization: float = 0.89,
) -> tuple[GenerateFn, dict[str, Any]]:  # fmt: skip
    """vLLM offline engine; returns (generate, extra-info callable result holder)."""
    from vllm import LLM, SamplingParams

    from regslm.teacher.engine import CHAT_TEMPLATE_KWARGS

    kw: dict[str, Any] = {
        "model": str(model), "max_model_len": max_len, "seed": 0,
        "gpu_memory_utilization": gpu_memory_utilization, "enforce_eager": True,
        "enable_prefix_caching": True, "attention_backend": "TRITON_ATTN",
    }  # fmt: skip
    if fmt == "fp8-online":
        kw["quantization"] = "fp8"
    if spec_draft is not None:
        kw["speculative_config"] = {
            "model": str(spec_draft),
            "num_speculative_tokens": spec_k,
            "method": "draft_model",
        }
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=0)
    info: dict[str, Any] = {"engine_kwargs": {k: v for k, v in kw.items()}}

    def gen(batch: Sequence[Messages], concurrency: int) -> list[Gen]:
        if not batch:
            return []
        if concurrency <= 1:
            outs = [
                llm.chat([m], sp, chat_template_kwargs=CHAT_TEMPLATE_KWARGS, use_tqdm=False)[0]
                for m in batch
            ]
            walls = None
        else:
            outs = llm.chat(list(batch), sp, chat_template_kwargs=CHAT_TEMPLATE_KWARGS)
            walls = True
        res = []
        for o in outs:
            m = getattr(o, "metrics", None)
            fin, arr = getattr(m, "finished_time", None), getattr(m, "arrival_time", None)
            lat = fin - arr if fin and arr else 0.0
            c = o.outputs[0]
            res.append(Gen(c.text, len(o.prompt_token_ids or []), len(c.token_ids), lat))
        if walls is None:  # single-stream: latency = wall of each call, measured by caller
            pass
        return res

    def spec_metrics() -> dict[str, Any]:
        try:
            ms = {m.name: m for m in llm.get_metrics()}
            drafts = ms["vllm:spec_decode_num_drafts"].value
            acc = ms["vllm:spec_decode_num_accepted_tokens"].value
            dtok = ms["vllm:spec_decode_num_draft_tokens"].value
            return {"num_drafts": drafts, "accepted": acc, "draft_tokens": dtok,
                    "acceptance_rate": acc / dtok if dtok else None}  # fmt: skip
        except Exception as e:  # metrics API differs across vLLM versions
            return {"error": repr(e)}

    info["spec_metrics"] = spec_metrics
    return gen, info


def _timed_single(generate: GenerateFn) -> GenerateFn:
    """Make single-stream latency a wall-clock measurement even if the engine reports none."""

    def g(batch: Sequence[Messages], concurrency: int) -> list[Gen]:
        if concurrency > 1:
            return generate(batch, concurrency)
        out: list[Gen] = []
        for m in batch:
            t0 = time.perf_counter()
            (r,) = generate([m], 1)
            out.append(Gen(r.text, r.n_prompt, r.n_gen, time.perf_counter() - t0))
        return out

    return g


def bench(
    run_id: str,
    backend: str,
    fmt: str,
    *,
    sets: str = "test",
    limit: int | None = None,
    threads: int = 6,
    concurrency: int | None = None,
    spec_draft_run: str | None = None,
    spec_k: int = 3,
    refs: str = "results/runs/silver-*/labels.jsonl",
    runs: Path = RUNS_DIR,
    out_root: Path = Path("results/bench"),
    n_boot: int = 1000,
    needle: bool = False,
    needle_ctx: int = 17408,
) -> dict[str, Any]:
    from regslm.train.evaluate import evaluate

    if backend not in BACKENDS or fmt not in FORMATS:
        raise ValueError(f"backend {BACKENDS}, format {FORMATS}")
    cfg = TrainConfig.load(run_id, runs)
    items = load_set(sets, cfg, refs.split(","), limit)
    if not items:
        raise RuntimeError(f"no scorable units in set {sets!r}")
    sub = f"{run_id}/bench/{backend}-{fmt}" + (f"-spec{spec_k}" if spec_draft_run else "")
    sub_dir = runs / sub
    gpu_csv = sub_dir / "gpu.csv"
    result: dict[str, Any] = {"run_id": run_id, "backend": backend, "format": fmt, "set": sets}
    ctx_per_slot = cfg.max_seq_len + MAX_NEW_TOKENS
    if needle:  # one engine/server serves both the protocol and the needle grid
        ctx_per_slot = max(ctx_per_slot, needle_ctx)
    extra: dict[str, Any] = {}
    handle: ServerHandle | None = None
    if backend == "vllm":
        if fmt.startswith("gguf"):
            raise ValueError("GGUF formats run on llamacpp backends")
        path = model_path(cfg, fmt, runs)
        draft = model_dir(TrainConfig.load(spec_draft_run, runs), runs) if spec_draft_run else None
        sub_dir.mkdir(parents=True, exist_ok=True)
        smi = subprocess.Popen(
            f"nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > {gpu_csv}",
            shell=True,
        )
        try:
            gen, extra = vllm_generate_fn(path, fmt, ctx_per_slot, draft, spec_k)
            proto = run_protocol(_timed_single(gen), items, concurrency or 24)
            timed = _timed_single(gen)
            if needle:
                result["needle"] = needle_phase(timed, cfg, refs, ctx_per_slot)
            extra = {"spec": extra["spec_metrics"]()} if draft else {}
        finally:
            smi.terminate()
        result["memory"] = {f"vram_{k}": v for k, v in vram_peak(gpu_csv).items()}
        result["model_size_gb"] = dir_size_gb(path)
    else:
        if not fmt.startswith("gguf"):
            raise ValueError("llama.cpp backends need a gguf-* format")
        gpu = backend == "llamacpp-cuda"
        gguf = gguf_path(cfg, fmt, runs)
        conc = concurrency or (4 if gpu else 1)
        sub_dir.mkdir(parents=True, exist_ok=True)
        smi = (
            subprocess.Popen(
                f"nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > {gpu_csv}",
                shell=True,
            )
            if gpu
            else None
        )
        try:
            with llama_server(
                gguf, gpu=gpu, threads=threads, ctx=ctx_per_slot * conc, parallel=conc,
                log=sub_dir / "llama-server.log",
            ) as handle:  # fmt: skip
                proto = run_protocol(handle.client.generate, items, conc)
                if needle:
                    result["needle"] = needle_phase(handle.client.generate, cfg, refs, ctx_per_slot)
                handle.sample_rss()
        finally:
            if smi:
                smi.terminate()
        result["threads"] = threads
        result["memory"] = {
            "host_ram_peak_mib": (handle.peak_rss_kib / 1024)
            if handle and handle.peak_rss_kib
            else None
        }
        if gpu:
            result["memory"].update({f"vram_{k}": v for k, v in vram_peak(gpu_csv).items()})
        result["model_size_gb"] = dir_size_gb(gguf)
        result["server"] = {"binary": str(LLAMA_CUDA_BIN if gpu else LLAMA_CPU_BIN)}
    write_predictions(
        sub_dir / "predictions" / f"{sets}.jsonl", sets, proto["items"], proto["gens"]
    )
    result["throughput"] = proto["throughput"]
    result["single_stream"] = proto["single_stream"]
    result.update(extra)
    result["eval"] = evaluate(sub, [sets], n_boot, runs)["sets"][sets]
    write_result(
        out_root / run_id, backend, fmt + (f"-spec{spec_k}" if spec_draft_run else ""), result
    )
    return result


def needle_phase(
    generate: GenerateFn,
    cfg: TrainConfig,
    refs: str,
    max_ctx: int,
    lengths: Sequence[int] = LENGTHS,
    depths: Sequence[float] = DEPTHS,
    n_targets: int = 3,
) -> dict[str, Any]:
    """Needle-in-rulebook grid on the largest held-out document; skips lengths > ``max_ctx``."""
    import polars as pl
    from transformers import AutoTokenizer

    from regslm.corpus.base import Unit

    items = [i for i in load_set("heldout", cfg, refs.split(",")) if i.reference]
    items = [i for i in items if i.reference and i.reference.get("obligations")]
    if not items:
        return {"error": "no heldout units with a reference obligation"}
    tok = AutoTokenizer.from_pretrained(cfg.base)

    class _Tok:
        def encode(self, text: str) -> list[int]:
            return list(tok(text, add_special_tokens=False)["input_ids"])

        def decode(self, ids: list[int]) -> str:
            return str(tok.decode(ids))

    by_doc: dict[str, list[Item]] = {}
    for it in items:
        by_doc.setdefault(it.unit.doc_id, []).append(it)
    doc_id = max(by_doc, key=lambda d: len(by_doc[d]))
    df = pl.read_parquet("data/corpus/units.parquet").filter(pl.col("doc_id") == doc_id)
    doc = [Unit.model_validate(r) for r in df.select(list(Unit.model_fields)).iter_rows(named=True)]
    targets = by_doc[doc_id][:n_targets]
    cells: list[dict[str, Any]] = []
    for length in lengths:
        if length + MAX_NEW_TOKENS > max_ctx:
            cells.append({"length": length, "skipped": f"context {max_ctx} too small"})
            continue
        for depth in depths:
            for it in targets:
                try:
                    n = build_context(doc, it.unit, _Tok(), length, depth)
                except ValueError as e:
                    cells.append({"length": length, "depth": depth, "skipped": str(e)})
                    continue
                (g,) = generate([n.messages], 1)
                sc = score_needle(g.text, it.unit, it.reference or {})
                cells.append({"length": length, "depth": depth, "unit_id": it.unit.unit_id,
                              "n_tokens": n.n_tokens, "latency_s": g.latency_s, **sc})  # fmt: skip
    return {"doc_id": doc_id, "cells": cells}


def llama_cpp_commit(root: Path = THIRD_PARTY / "llama.cpp") -> str | None:
    git = shutil.which("git")
    if not git or not root.exists():
        return None
    r = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None
