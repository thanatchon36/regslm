"""``regslm extract``: document -> units -> model (vLLM or llama.cpp) -> resolved obligations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from regslm.corpus.base import Unit
from regslm.rewards.components import resolved
from regslm.train.prompts import student_messages

DISCLAIMER = "machine extraction — review by a qualified person before any use"
MAX_NEW_TOKENS = 1024
BACKENDS = ("vllm", "llamacpp")
FORMATS = ("bf16", "awq", "gguf")
Messages = list[dict[str, str]]


class Backend(Protocol):
    def generate(self, batch: Sequence[Messages]) -> list[str]: ...


class FnBackend:
    """Backend from a plain callable (tests, custom runtimes)."""

    def __init__(self, fn: Callable[[Sequence[Messages]], list[str]]) -> None:
        self.fn = fn

    def generate(self, batch: Sequence[Messages]) -> list[str]:
        return self.fn(batch)


# ---------------------------------------------------------------- backends


class VllmBackend:
    def __init__(self, model: str, guided: bool = False, max_model_len: int = 4096) -> None:
        from vllm import SamplingParams

        from regslm.teacher.engine import EngineConfig, build_llm, build_sampling

        cfg = EngineConfig(model=model, max_model_len=max_model_len, max_tokens=MAX_NEW_TOKENS)
        self.llm = build_llm(cfg)
        self.sampling = (
            build_sampling(cfg)
            if guided
            else SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=cfg.seed)
        )

    def generate(self, batch: Sequence[Messages]) -> list[str]:
        from regslm.teacher.engine import CHAT_TEMPLATE_KWARGS

        outs = self.llm.chat(list(batch), self.sampling, chat_template_kwargs=CHAT_TEMPLATE_KWARGS)
        return [o.outputs[0].text for o in outs]


def resolve_gguf(model: str) -> Path:
    """A local ``.gguf`` file / directory, or a Hugging Face repo id holding GGUF files."""
    p = Path(model)
    if p.is_file():
        return p
    if p.is_dir():
        hits = sorted(p.glob("*Q4_K_M*.gguf")) or sorted(p.glob("*.gguf"))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"no .gguf file in {p}")
    from huggingface_hub import list_repo_files, snapshot_download

    files = [f for f in list_repo_files(model) if f.endswith(".gguf")]
    if not files:
        raise FileNotFoundError(f"no .gguf file in Hugging Face repo {model}")
    pick = next((f for f in files if "Q4_K_M" in f), files[0])
    return Path(snapshot_download(model, allow_patterns=[pick])) / pick


def llama_server_binary() -> Path:
    """Directory containing ``llama-server``: $LLAMA_SERVER_DIR, PATH, or third_party builds."""
    env = os.environ.get("LLAMA_SERVER_DIR")
    cands = [Path(env)] if env else []
    w = shutil.which("llama-server")
    if w:
        cands.append(Path(w).parent)
    from regslm.bench.runner import LLAMA_CPU_BIN, LLAMA_CUDA_BIN

    cands += [LLAMA_CUDA_BIN, LLAMA_CPU_BIN]
    for d in cands:
        if (d / "llama-server").exists():
            return d.resolve()
    raise FileNotFoundError("llama-server not found; set LLAMA_SERVER_DIR or put it on PATH")


class LlamaBackend:
    """Starts ``llama-server`` on a free port; ``close()`` stops it."""

    def __init__(
        self, gguf: Path, *, gpu: bool = False, threads: int = 6, ctx: int = 4096 * 4,
        parallel: int = 4,
    ) -> None:  # fmt: skip
        from regslm.bench.runner import LlamaClient, free_port, llama_server_cmd

        binary, port = llama_server_binary(), free_port()
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{binary}:{env.get('LD_LIBRARY_PATH', '')}"
        self.proc = subprocess.Popen(
            llama_server_cmd(binary, gguf, port, gpu=gpu, threads=threads, ctx=ctx,
                             parallel=parallel),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env,
        )  # fmt: skip
        self.parallel = parallel
        self.client = LlamaClient(f"http://127.0.0.1:{port}")
        try:
            self.client.wait_ready(alive=lambda: self.proc.poll() is None)
        except BaseException:
            self.close()
            raise

    def generate(self, batch: Sequence[Messages]) -> list[str]:
        return [g.text for g in self.client.generate(batch, self.parallel)]

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def choose_backend(backend: str | None, fmt: str | None, model: str) -> tuple[str, str]:
    """Resolve (backend, format) defaults and reject inconsistent pairs."""
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    if fmt is not None and fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    if backend is None:
        backend = "llamacpp" if fmt == "gguf" or model.lower().endswith(".gguf") else "vllm"
    if fmt is None:
        fmt = "gguf" if backend == "llamacpp" else ("awq" if "awq" in model.lower() else "bf16")
    if (backend == "llamacpp") != (fmt == "gguf"):
        raise ValueError("llamacpp needs --format gguf; vllm needs --format bf16 or awq")
    return backend, fmt


@contextmanager
def open_backend(
    model: str, backend: str, guided: bool = False, gpu: bool = False, threads: int = 6
) -> Iterator[Backend]:
    if backend == "vllm":
        yield VllmBackend(model, guided=guided)
        return
    if guided:
        raise ValueError("--guided is only supported with the vllm backend")
    b = LlamaBackend(resolve_gguf(model), gpu=gpu, threads=threads)
    try:
        yield b
    finally:
        b.close()


# ---------------------------------------------------------------- extraction


def extract_units(
    units: Sequence[Unit], backend: Backend, batch_size: int = 64
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the model over units; return obligation rows and summary counts."""
    rows: list[dict[str, Any]] = []
    counts = {"units": len(units), "obligations": 0, "invalid": 0, "unresolved": 0}
    invalid_ids: list[str] = []
    for i in range(0, len(units), batch_size):
        chunk = units[i : i + batch_size]
        completions = backend.generate([student_messages(u) for u in chunk])
        for u, c in zip(chunk, completions, strict=True):
            r = resolved(c, u)
            if r is None:
                counts["invalid"] += 1
                invalid_ids.append(u.unit_id)
                continue
            counts["unresolved"] += len(r.unresolved)
            for o in r.extraction.obligations:
                row = o.model_dump(mode="json")
                span = o.citation.span
                row["quote"] = u.text[span.start : span.end]
                row["unit_text"] = u.text
                rows.append(row)
            counts["obligations"] += len(r.extraction.obligations)
    return rows, {**counts, "invalid_unit_ids": invalid_ids}


def write_jsonl(rows: Sequence[dict[str, Any]], summary: dict[str, Any], out: Any) -> None:
    for r in rows:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.write(json.dumps({"_summary": {**summary, "disclaimer": DISCLAIMER}}, ensure_ascii=False))
    out.write("\n")


def summary_line(s: dict[str, Any]) -> str:
    return (
        f"{s['units']} units, {s['obligations']} obligations, {s['invalid']} invalid, "
        f"{s['unresolved']} unresolved quotes. {DISCLAIMER}"
    )
