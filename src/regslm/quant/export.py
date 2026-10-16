"""Quantised exports of a merged checkpoint (docs/specs/quant_bench.md, paper §4.6).

Runs in ``.venv-quant`` (llmcompressor). Each format runs in a child process so that wall time
and peak host RAM (``wait4`` rusage) are per format; peak VRAM comes from torch inside the
child. Failures are recorded, not raised. Tool versions land in ``quant.json``.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from regslm.train.config import RUNS_DIR, TrainConfig

FORMATS = ("fp8", "awq", "gptq", "gguf-q8", "gguf-q4km")
GGUF_TYPES = {"gguf-q8": "Q8_0", "gguf-q4km": "Q4_K_M"}
LLAMA_ROOT = Path("third_party/llama.cpp")
CALIB_N = int(os.environ.get("REGSLM_CALIB_N", "256"))
CALIB_MAX_LEN = 2048


def parse_formats(s: str) -> list[str]:
    fs = [f.strip() for f in s.split(",") if f.strip()]
    bad = [f for f in fs if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown format(s) {bad}; choose from {FORMATS}")
    return fs


def merged_dir(cfg: TrainConfig, runs: Path = RUNS_DIR) -> Path:
    """bf16 merged checkpoint; ``merged_bf16`` when ``merged`` is an FP8 serving copy (D-050)."""
    ck = cfg.run_dir(runs) / "checkpoints"
    d = ck / "merged_bf16" if (ck / "merged_bf16" / "config.json").exists() else ck / "merged"
    if not (d / "config.json").exists():
        raise FileNotFoundError(f"no merged checkpoint at {d}; train first")
    return d


# ------------------------------------------------------------------------- calibration


def select_calibration_units(
    units: dict[str, Any], gold_ids: set[str], n: int = CALIB_N, seed: int = 0
) -> list[Any]:
    """``n`` train units in a fixed hash order, never a gold unit (never test/dev/heldout)."""
    from regslm.train.data import hash_order

    ids = [i for i in hash_order(sorted(units), seed) if i not in gold_ids]
    return [units[i] for i in ids[:n]]


def calibration_texts(cfg: TrainConfig, tokenizer: Any, n: int = CALIB_N) -> list[str]:
    """Student-prompt rendering (chat template, generation prompt) of ``n`` train units."""
    from regslm.data.splits import read_gold_ids
    from regslm.train.data import load_train_units, render_prompt
    from regslm.train.prompts import student_messages

    if cfg.splits is None:
        raise ValueError("calibration needs data/splits (train units); fixture mode has none")
    units = load_train_units(cfg.splits)
    chosen = select_calibration_units(units, read_gold_ids(Path(cfg.gold_sample)), n)
    return [render_prompt(tokenizer, student_messages(u)) for u in chosen]


# ---------------------------------------------------------------------------- recipes


def recipe(fmt: str) -> Any:
    """llmcompressor modifier for a format (W4A16 presets are group size 128)."""
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    ignore = ["lm_head"]
    if fmt == "fp8":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)
    if fmt == "awq":
        return AWQModifier(targets=["Linear"], scheme="W4A16", ignore=ignore)
    if fmt == "gptq":
        return GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
    raise ValueError(fmt)


def export_llmcompressor(fmt: str, src: Path, out: Path, texts: Sequence[str]) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype="auto")
    kw: dict[str, Any] = {}
    if fmt != "fp8":  # FP8_DYNAMIC needs no calibration data
        ds = Dataset.from_dict({"text": list(texts)}).map(
            lambda r: tok(r["text"], max_length=CALIB_MAX_LEN, truncation=True,
                          add_special_tokens=False),
            remove_columns=["text"],
        )  # fmt: skip
        kw = {"dataset": ds, "max_seq_length": CALIB_MAX_LEN, "num_calibration_samples": len(texts)}
    oneshot(model=model, recipe=recipe(fmt), **kw)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    return {"peak_vram_mib": peak, "n_calibration": len(texts)}


# -------------------------------------------------------------------------------- GGUF


def convert_cmd(src: Path, out: Path, llama: Path = LLAMA_ROOT) -> list[str]:
    return [sys.executable, str(llama / "convert_hf_to_gguf.py"), str(src), "--outfile", str(out),
            "--outtype", "f16"]  # fmt: skip


def quantize_cmd(f16: Path, out: Path, qtype: str, llama: Path = LLAMA_ROOT) -> list[str]:
    return [str(llama / "build" / "bin" / "llama-quantize"), str(f16), str(out), qtype]


def export_gguf(
    fmts: Sequence[str], src: Path, out_dir: Path, llama: Path = LLAMA_ROOT, keep_f16: bool = False
) -> dict[str, dict[str, Any]]:
    """f16 GGUF once, then one quantised file per format. Returns per-format info."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / "model-f16.gguf"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    res: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    if not f16.exists():
        subprocess.run(convert_cmd(src, f16, llama), check=True, env=env)
    t_conv = time.time() - t0
    for f in fmts:
        t1 = time.time()
        qt = GGUF_TYPES[f]
        out = out_dir / f"model-{qt}.gguf"
        bin_dir = (llama / "build" / "bin").resolve()
        subprocess.run(
            quantize_cmd(f16, out, qt, llama), check=True,
            env={**env, "LD_LIBRARY_PATH": f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"},
        )  # fmt: skip
        res[f] = {"file": out.name, "convert_s": t_conv, "quantize_s": time.time() - t1}
    if not keep_f16:
        f16.unlink(missing_ok=True)
    return res


# ------------------------------------------------------------------------ orchestration


def dir_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def tool_versions(llama: Path = LLAMA_ROOT) -> dict[str, Any]:
    from importlib import metadata

    v: dict[str, Any] = {"python": sys.version.split()[0]}
    for pkg in ("llmcompressor", "compressed-tensors", "torch", "transformers", "gguf", "vllm"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            v[pkg] = metadata.version(pkg)
    if llama.exists():
        r = subprocess.run(
            ["git", "-C", str(llama), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        v["llama.cpp_commit"] = r.stdout.strip() or None
    return v


@dataclass
class Job:
    fmts: list[str]  # one llmcompressor format, or all requested gguf formats


def plan_jobs(fmts: Sequence[str]) -> list[Job]:
    jobs = [Job([f]) for f in fmts if not f.startswith("gguf")]
    g = [f for f in fmts if f.startswith("gguf")]
    return jobs + ([Job(g)] if g else [])


def worker(run_id: str, fmts: list[str], runs: Path = RUNS_DIR) -> dict[str, dict[str, Any]]:
    """Runs in the child: export + measure; returns per-format records (errors captured)."""
    cfg = TrainConfig.load(run_id, runs)
    src = merged_dir(cfg, runs)
    qdir = cfg.run_dir(runs) / "quant"
    t0 = time.time()
    out: dict[str, dict[str, Any]] = {}
    try:
        if fmts[0].startswith("gguf"):
            out = export_gguf(fmts, src, qdir / "gguf")
            for f in fmts:
                out[f]["size_bytes"] = (qdir / "gguf" / out[f]["file"]).stat().st_size
        else:
            from transformers import AutoTokenizer

            fmt = fmts[0]
            texts = (
                [] if fmt == "fp8" else calibration_texts(cfg, AutoTokenizer.from_pretrained(src))
            )
            out[fmt] = export_llmcompressor(fmt, src, qdir / fmt, texts)
            out[fmt]["size_bytes"] = dir_bytes(qdir / fmt)
        for f in fmts:
            out[f]["status"] = "ok"
    except Exception as e:  # failures are results
        for f in fmts:
            out.setdefault(f, {}).update(status="failed", error=f"{type(e).__name__}: {e}")
    for r in out.values():
        r["wall_s"] = time.time() - t0
    return out


def quantize(run_id: str, fmts: Sequence[str], runs: Path = RUNS_DIR) -> dict[str, Any]:
    """Parent: one child per job; merges results into ``results/runs/<id>/quant.json``."""
    cfg = TrainConfig.load(run_id, runs)
    merged_dir(cfg, runs)
    run_dir = cfg.run_dir(runs)
    qj = run_dir / "quant.json"
    rec: dict[str, Any] = json.loads(qj.read_text()) if qj.exists() else {"formats": {}}
    rec["run_id"], rec["tools"] = run_id, tool_versions()
    for job in plan_jobs(fmts):
        tmp = run_dir / f".quant-{'-'.join(job.fmts)}.json"
        cmd = [sys.executable, "-m", "regslm.quant.export", run_id, str(runs), str(tmp), *job.fmts]
        proc = subprocess.Popen(cmd)
        _, status, ru = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        res: dict[str, Any] = json.loads(tmp.read_text()) if tmp.exists() else {}
        tmp.unlink(missing_ok=True)
        for f in job.fmts:
            r = res.get(f) or {"status": "failed", "error": f"worker exited {proc.returncode}"}
            r["peak_host_ram_mib"] = ru.ru_maxrss / 1024  # Linux: KiB
            rec["formats"][f] = r
        qj.write_text(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":  # child entry: <run_id> <runs_dir> <out.json> <fmt>...
    _run, _runs, _out, *_fmts = sys.argv[1:]
    Path(_out).write_text(json.dumps(worker(_run, _fmts, Path(_runs))))
