"""SFT trainer (Unsloth -> TRL/PEFT fallback), docs/specs/train.md "Trainer". Heavy imports lazy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from regslm.train.config import RUNS_DIR, TrainConfig, total_steps

LOSS_CURVE_POINTS = 60


def is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)


def loss_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    pts = [(int(h["step"]), float(h["loss"])) for h in history if "loss" in h and "step" in h]
    if not pts:
        return {"n_logged": 0}
    losses = [v for _, v in pts]
    tail = losses[-max(1, len(losses) // 10) :]
    stride = max(1, math.ceil(len(pts) / LOSS_CURVE_POINTS))
    return {
        "n_logged": len(pts),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "mean_last_10pct": sum(tail) / len(tail),
        "curve": pts[::stride],
    }


def snapshot_info(base: str, revision: str | None) -> dict[str, str | None]:
    """Base revision sha (HF cache snapshot dir) and sha256 of tokenizer.json."""
    try:
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(base, revision=revision, local_files_only=True))
    except Exception:
        return {"base_revision_sha": revision, "tokenizer_sha256": None}
    tj = d / "tokenizer.json"
    return {
        "base_revision_sha": d.name,
        "tokenizer_sha256": hashlib.sha256(tj.read_bytes()).hexdigest() if tj.exists() else None,
    }


def _load_model(cfg: TrainConfig) -> tuple[Any, Any, str]:
    """Return (model, tokenizer, backend). Unsloth first; plain TRL/PEFT otherwise."""
    if cfg.backend in ("auto", "unsloth"):
        try:
            from unsloth import FastLanguageModel  # must precede transformers/peft/trl

            model, tok = FastLanguageModel.from_pretrained(
                model_name=cfg.base,
                max_seq_length=cfg.max_seq_len,
                dtype=None,
                load_in_4bit=cfg.method == "qlora",
                full_finetuning=cfg.method == "full",
                revision=cfg.revision,
            )
            if cfg.method != "full":
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=cfg.lora_r,
                    target_modules=cfg.target_modules,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    bias="none",
                    use_gradient_checkpointing="unsloth" if cfg.grad_ckpt else False,
                    random_state=cfg.seed,
                )
            return model, getattr(tok, "tokenizer", tok), "unsloth"
        except Exception as e:  # base unsupported / kernel missing -> record and fall back
            if is_oom(e) or cfg.backend == "unsloth":
                raise
            print(f"[train] Unsloth unavailable ({type(e).__name__}: {e}); using plain TRL/PEFT")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw: dict[str, Any] = {"revision": cfg.revision, "dtype": torch.bfloat16}
    if cfg.method == "qlora":
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.base, device_map={"": 0}, **kw)
    tok = AutoTokenizer.from_pretrained(cfg.base, revision=cfg.revision)
    if cfg.method != "full":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.grad_ckpt)
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    return model, tok, "plain"


# D-050: vLLM 0.31 cannot load bitsandbytes checkpoints, and an 8B model does not fit 12 GB in
# bf16. Adapters trained on a pre-quantised bnb base are merged into its bf16 original; bases in
# SERVE_FP8 keep the bf16 merge in ``merged_bf16`` (input for RQ4 quantization) and are served from
# an FP8-dynamic copy in ``merged``.
MERGE_BASES = {"unsloth/Qwen3-8B-unsloth-bnb-4bit": "Qwen/Qwen3-8B"}
SERVE_FP8 = {
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "Qwen/Qwen3-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT",
}
REVISIONS_PATH = Path("results/base_revisions.json")


def merge_source(cfg: TrainConfig, revisions: Path = REVISIONS_PATH) -> tuple[str, str | None]:
    """(model id, revision) to merge the adapter into: the bf16 original of a bnb base."""
    src = MERGE_BASES.get(cfg.base)
    if src is None:
        return cfg.base, cfg.revision
    rev = json.loads(revisions.read_text()).get(src) if revisions.exists() else None
    if not rev:
        raise ValueError(f"merge base {src} has no pinned revision in {revisions}")
    return src, rev


def fp8_convert(src: Path, out: Path) -> None:
    """FP8-dynamic copy of a bf16 checkpoint (llmcompressor lives in .venv-quant)."""
    import subprocess

    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": ".venv-quant"}
    cmd = ["uv", "run", "--no-sync", "regslm", "fp8", "--src", str(src), "--out", str(out)]
    subprocess.run(cmd, env=env, check=True)


def merge_adapter(cfg: TrainConfig, adapter_dir: Path, out_dir: Path) -> None:
    """Merge a LoRA adapter into the bf16 base on CPU (also for QLoRA) and save; FP8 copy for
    SERVE_FP8 bases (D-050)."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src, rev = merge_source(cfg)
    fp8 = cfg.base in SERVE_FP8
    dest = out_dir.parent / "merged_bf16" if fp8 else out_dir
    base = AutoModelForCausalLM.from_pretrained(src, revision=rev, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(dest, safe_serialization=True)
    AutoTokenizer.from_pretrained(src, revision=rev).save_pretrained(dest)
    del base, merged
    gc.collect()
    if fp8:
        fp8_convert(dest, out_dir)


def _train(cfg: TrainConfig, run_dir: Path) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t0 = time.time()
    model, tok, backend = _load_model(cfg)
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    from regslm.train.data import build_from_config, completion_suffix, encode

    set_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    built, _held = build_from_config(cfg, tok)
    if not built.examples:
        raise RuntimeError(f"no training examples: {built.report}")
    suffix = completion_suffix(tok)
    with (run_dir / "train_examples.jsonl").open("w", encoding="utf-8") as f:
        for e in built.examples:
            row = {"unit_id": e.unit_id, "origin": e.origin, "target": e.target}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ds = Dataset.from_list([encode(tok, e, suffix) for e in built.examples])
    steps = total_steps(len(ds), cfg)
    eff = cfg.per_device_batch * cfg.grad_accum
    ckpt = run_dir / "checkpoints"
    args = SFTConfig(
        output_dir=str(ckpt / "trainer"),
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.effective_lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(0.03 * steps)),
        max_steps=steps,
        bf16=cfg.bf16,
        optim=cfg.effective_optim,
        seed=cfg.seed,
        data_seed=cfg.seed,
        logging_steps=max(1, steps // 100),
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=cfg.grad_ckpt,
        max_length=cfg.max_seq_len,
        completion_only_loss=True,
        packing=False,
        weight_decay=0.0,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.train()
    nondet = sorted(
        {
            str(w.message).splitlines()[0][:160]
            for w in caught
            if "determin" in str(w.message).lower()
        }
    )
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0

    lens = [len(x) for x in ds["input_ids"]]
    seen = int(sum(lens) / len(lens) * steps * eff)
    out = ckpt / ("merged" if cfg.method == "full" else "adapter")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    metrics: dict[str, Any] = {
        "status": "ok",
        "trainer_backend": backend,
        "config_hash": cfg.hash(),
        "steps": steps,
        "epochs_effective": steps * eff / len(ds),
        "n_examples": len(ds),
        "tokens_seen": seen,
        "train_wall_s": train_s,
        "peak_vram_allocated_bytes": peak,
        "peak_vram_reserved_bytes": reserved,
        "nondeterministic_ops": nondet,
        "loss": loss_summary(trainer.state.log_history),
        "data": built.report,
        **snapshot_info(cfg.base, cfg.revision),
    }
    del trainer, model
    torch.cuda.empty_cache()
    if cfg.method != "full":
        merge_adapter(cfg, out, ckpt / "merged")
    metrics["wall_s"] = time.time() - t0
    return metrics


def run_training(
    cfg: TrainConfig,
    runs: Path = RUNS_DIR,
    trainer: Callable[[TrainConfig, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train and write ``metrics.json``. OOM is a result (``does_not_fit``), exit code 0."""
    run_dir = cfg.run_dir(runs)
    run_dir.mkdir(parents=True, exist_ok=True)
    fn = trainer or _train
    t0 = time.time()
    try:
        metrics = fn(cfg, run_dir)
    except Exception as e:
        if not is_oom(e):
            raise
        metrics = {
            "status": "does_not_fit",
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "config_hash": cfg.hash(),
            "base": cfg.base,
            "method": cfg.method,
            "wall_s": time.time() - t0,
        }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
