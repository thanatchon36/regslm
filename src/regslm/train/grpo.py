"""GRPO with verifiable rewards (docs/specs/grpo.md). Heavy imports (torch/trl) are lazy.

Run layout: ``config.yaml`` (a ``TrainConfig`` copied from the parent so ``regslm predict`` works)
plus ``grpo.yaml`` (``GrpoConfig``); outputs ``metrics.json``, ``reward_components.jsonl``,
``hacking.json``, ``checkpoints/{adapter,merged}``.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from regslm.corpus.base import Unit
from regslm.rewards import batch_stats, load_weights, resolved, reward
from regslm.schema.obligation import Extraction
from regslm.train.config import RUNS_DIR, TrainConfig
from regslm.train.data import LabelRow, hash_order, read_label_rows
from regslm.train.prompts import student_messages
from regslm.train.sft import is_oom, snapshot_info

EMPTY_PP = 0.10  # RQ3: empty-rate rise (absolute) that flags hacking
RECALL_PP = 0.05  # RQ3: recall drop (absolute) while total reward rises


class GrpoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    parent: str  # SFT run id (checkpoints/adapter or checkpoints/merged)
    n_prompts: int = 2000
    numeric_frac: float = 0.2
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.04  # KL coefficient
    lr: float = 5e-6
    warmup_frac: float = 0.05
    prompts_per_device: int = 2
    prompts_per_step: int = 16
    max_steps: int = 250
    save_steps: int = 50
    eval_steps: list[int] = Field(default_factory=lambda: [0, 100, 250])
    eval_n: int = 300
    max_wall_s: float = 3 * 3600
    reward_weights: str = "config/rewards.yaml"
    loss_type: str | None = None  # None -> TRL default (recorded in metrics.json)
    scale_rewards: str | None = None  # None -> TRL default (recorded)
    seed: int = 0
    backend: Literal["auto", "unsloth", "plain"] = "auto"
    lora_r: int = 16  # fresh LoRA, only for full-FT parents
    lora_alpha: int = 32
    bf16: bool = True
    grad_ckpt: bool = True
    retry_g2: bool = True  # on OOM retry once as separate cell `<id>-g2` with 2 generations

    def run_dir(self, runs: Path = RUNS_DIR) -> Path:
        return runs / self.run_id

    def save(self, runs: Path = RUNS_DIR) -> Path:
        p = self.run_dir(runs) / "grpo.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
        return p

    @classmethod
    def load(cls, run_id: str, runs: Path = RUNS_DIR) -> GrpoConfig:
        return cls.model_validate(yaml.safe_load((runs / run_id / "grpo.yaml").read_text()))


def default_num_generations(base: str) -> int:
    return 4 if "4b" in base.lower() else 8


def write_run_config(
    run_id: str, parent: str, overrides: Mapping[str, Any] | None = None, runs: Path = RUNS_DIR
) -> GrpoConfig:
    """Write ``config.yaml`` (parent TrainConfig, new run id) and ``grpo.yaml``."""
    pcfg = TrainConfig.load(parent, runs)
    d: dict[str, Any] = {
        "run_id": run_id, "parent": parent,
        "num_generations": default_num_generations(pcfg.base), "seed": pcfg.seed,
    }  # fmt: skip
    d.update(overrides or {})
    gcfg = GrpoConfig.model_validate(d)
    pcfg.model_copy(update={"run_id": run_id}).save(runs)
    gcfg.save(runs)
    return gcfg


def select_parent(base: str, runs: Path = RUNS_DIR) -> str:
    """Best SFT run for ``base``: highest silver_val OF1 (never test/gold); ties -> smaller N."""
    best: tuple[float, int, str] | None = None
    for d in sorted(runs.glob("*/")):
        ev, cf = d / "eval.json", d / "config.yaml"
        if not (ev.exists() and cf.exists()) or (d / "grpo.yaml").exists():
            continue
        try:
            cfg = TrainConfig.load(d.name, runs)
            of1 = float(json.loads(ev.read_text())["sets"]["silver_val"]["metrics"]["of1"])
        except Exception:
            continue
        if cfg.base != base:
            continue
        cand = (of1, -cfg.n_labels, d.name)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        raise LookupError(f"no evaluated SFT run on silver_val for {base}")
    return best[2]


# ------------------------------------------------------------------------------ prompts


def is_numeric(ex: Extraction) -> bool:
    return any(o.threshold is not None or o.deadline for o in ex.obligations)


def sample_prompts(
    rows: Mapping[str, LabelRow],
    units: Mapping[str, Unit],
    sft_ids: set[str],
    n: int,
    numeric_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    """Pick ``n`` unit ids: ``numeric_frac`` with a threshold/deadline, preferring units outside
    the parent's SFT subset. Reports the overlap fraction with that subset."""
    ok = [
        u
        for u in hash_order(rows, seed)
        if u in units and rows[u].valid and rows[u].extraction is not None
        and rows[u].n_unresolved == 0
    ]  # fmt: skip
    num = {u for u in ok if is_numeric(rows[u].extraction)}  # type: ignore[arg-type]
    fresh = [u for u in ok if u not in sft_ids]
    seen = [u for u in ok if u in sft_ids]
    chosen: list[str] = []

    def take(cands: Sequence[str], k: int) -> None:
        have = set(chosen)
        for u in cands:
            if k <= 0 or len(chosen) >= n:
                return
            if u not in have:
                chosen.append(u)
                k -= 1

    n_num = min(n, round(n * numeric_frac))
    take([u for u in fresh if u in num] + [u for u in seen if u in num], n_num)
    take([u for u in fresh if u not in num] + [u for u in seen if u not in num], n - len(chosen))
    take(fresh + seen, n - len(chosen))  # numeric leftovers if non-numeric ran out
    picked = hash_order(chosen, seed + 1)
    n_sel = len(picked)
    report = {
        "n_eligible": len(ok),
        "n_selected": n_sel,
        "n_numeric": sum(u in num for u in picked),
        "numeric_share": sum(u in num for u in picked) / n_sel if n_sel else 0.0,
        "n_in_sft_subset": sum(u in sft_ids for u in picked),
        "overlap_fraction": sum(u in sft_ids for u in picked) / n_sel if n_sel else 0.0,
    }
    return picked, report


def prompt_rows(
    ids: Sequence[str], units: Mapping[str, Unit], rows: Mapping[str, LabelRow]
) -> list[dict[str, Any]]:
    """Dataset rows: conversational ``prompt`` plus ``unit_json``/``reference_json`` columns."""
    out = []
    for uid in ids:
        ex = rows[uid].extraction
        assert ex is not None
        out.append(
            {
                "prompt": student_messages(units[uid]),
                "unit_json": units[uid].model_dump_json(),
                "reference_json": ex.model_dump_json(),
            }
        )
    return out


# ------------------------------------------------------------------------------- reward


def completion_text(c: Any) -> str:
    """TRL passes a string (standard) or ``[{"role": "assistant", "content": ...}]`` (chat)."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(str(m.get("content", "")) for m in c if isinstance(m, dict))
    return str(c)


def make_reward_fn(
    weights: Mapping[str, float] | str | Path | None = None,
    log_path: Path | None = None,
    clock: Callable[[], float] = time.time,
) -> Callable[..., list[float]]:
    """TRL reward function ``(prompts, completions, unit_json, reference_json, **kw)``.

    ``unit_json``/``reference_json`` are dataset columns, already repeated per completion by TRL,
    so they align one-to-one with ``completions``. Logs per-call component means.
    """
    w = load_weights(weights)

    def reward_fn(
        prompts: Sequence[Any],
        completions: Sequence[Any],
        unit_json: Sequence[str],
        reference_json: Sequence[str],
        **kwargs: Any,
    ) -> list[float]:
        if not (len(completions) == len(unit_json) == len(reference_json)):
            raise ValueError("completions / unit_json / reference_json misaligned")
        texts = [completion_text(c) for c in completions]
        bds, n_obl = [], []
        for t, uj, rj in zip(texts, unit_json, reference_json, strict=True):
            unit = Unit.model_validate_json(uj)
            ref = Extraction.model_validate_json(rj)
            bds.append(reward(t, unit, ref, w))
            r = resolved(t, unit)
            n_obl.append(len(r.extraction.obligations) + len(r.unresolved) if r else 0)
        stats = batch_stats(texts, bds)
        stats["reward/mean_obligations"] = sum(n_obl) / max(1, len(n_obl))
        state = kwargs.get("trainer_state")
        if log_path is not None:
            rec = {"step": getattr(state, "global_step", None), "t": clock(), "n": len(texts)}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, **stats}) + "\n")
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            for k in ("reward/empty_frac", "reward/recall", "reward/mean_obligations"):
                log_metric(k, stats[k])
        return [b.total for b in bds]

    return reward_fn


# ------------------------------------------------------------------- hacking & budget


def hacking_flags(evals: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """RQ3 rule vs the step-0 eval: empty-rate +10 pp, or recall -5 pp while reward rises."""
    pts = sorted(evals, key=lambda e: e["step"])
    reasons: list[dict[str, Any]] = []
    if pts and pts[0]["step"] == 0:
        b = pts[0]
        for e in pts[1:]:
            if e["empty_frac"] - b["empty_frac"] >= EMPTY_PP - 1e-9:
                reasons.append({"step": e["step"], "rule": "empty_rate_up_10pp"})
            if b["recall"] - e["recall"] >= RECALL_PP - 1e-9 and e["reward"] > b["reward"]:
                reasons.append({"step": e["step"], "rule": "recall_down_5pp_reward_up"})
    return {"hacked": bool(reasons), "reasons": reasons, "evals": list(pts)}


class BudgetCallback:
    """Stop (and save) once wall time exceeds ``max_wall_s`` (duck-typed TrainerCallback)."""

    def __init__(self, max_wall_s: float, clock: Callable[[], float] = time.time) -> None:
        self.max_wall_s, self.clock = max_wall_s, clock
        self.t0 = clock()
        self.stopped = False

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
        if self.clock() - self.t0 > self.max_wall_s:
            control.should_training_stop = True
            control.should_save = True
            self.stopped = True
        return control


class EvalCallback:
    """Run ``eval_fn(model, step)`` at the configured steps (0 = train begin)."""

    def __init__(self, steps: Sequence[int], eval_fn: Callable[[Any, int], dict[str, float]]):
        self.steps, self.eval_fn = set(steps), eval_fn
        self.results: list[dict[str, float]] = []

    def _run(self, model: Any, step: int) -> None:
        if step in self.steps and all(r["step"] != step for r in self.results):
            self.results.append(self.eval_fn(model, step))

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any):
        self._run(model, int(state.global_step))
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **kw: Any):
        self._run(model, int(state.global_step))
        return control


def eval_summary(texts: Sequence[str], pairs: Sequence[tuple[Unit, Extraction]], w: Any) -> dict:
    bds = [reward(t, u, r, w) for t, (u, r) in zip(texts, pairs, strict=True)]
    s = batch_stats(texts, bds)
    return {
        "reward": s["reward/total"], "recall": s["reward/recall"],
        "empty_frac": s["reward/empty_frac"], "mean_len": s["reward/mean_len"],
        "schema_valid": s["reward/schema"],
    }  # fmt: skip


# -------------------------------------------------------------------------------- train


def _parent_dirs(pcfg: TrainConfig, runs: Path) -> tuple[Path, bool]:
    ck = pcfg.run_dir(runs) / "checkpoints"
    if pcfg.method == "full":
        return ck / "merged", True
    return ck / "adapter", False


def _load_model(cfg: GrpoConfig, pcfg: TrainConfig, runs: Path) -> tuple[Any, Any, str, str]:
    """(model, tokenizer, backend, base_for_merge). Continue the parent LoRA; fresh LoRA on the
    merged model for full-FT parents. Unsloth first, plain PEFT as fallback."""
    parent_dir, fresh = _parent_dirs(pcfg, runs)
    base = str(parent_dir) if fresh else pcfg.base
    if cfg.backend in ("auto", "unsloth"):
        try:
            from unsloth import FastLanguageModel

            model, tok = FastLanguageModel.from_pretrained(
                model_name=str(parent_dir), max_seq_length=pcfg.max_seq_len, dtype=None,
                load_in_4bit=False,
            )  # fmt: skip
            if fresh:
                model = FastLanguageModel.get_peft_model(
                    model, r=cfg.lora_r, target_modules=pcfg.target_modules,
                    lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
                    use_gradient_checkpointing="unsloth" if cfg.grad_ckpt else False,
                    random_state=cfg.seed,
                )  # fmt: skip
            return model, getattr(tok, "tokenizer", tok), "unsloth", base
        except Exception as e:
            if is_oom(e) or cfg.backend == "unsloth":
                raise
            print(f"[grpo] Unsloth unavailable ({type(e).__name__}: {e}); using plain TRL/PEFT")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = None if fresh else pcfg.revision
    model = AutoModelForCausalLM.from_pretrained(
        base, revision=rev, device_map={"": 0}, dtype=torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    if fresh:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=pcfg.target_modules,
                bias="none", task_type="CAUSAL_LM",
            ),
        )  # fmt: skip
    else:
        model = PeftModel.from_pretrained(model, str(parent_dir), is_trainable=True)
    return model, tok, "plain", base


def _pool(
    pcfg: TrainConfig, runs: Path
) -> tuple[dict[str, Unit], dict[str, LabelRow], dict[str, Unit], dict[str, LabelRow], set[str]]:
    """(train units, train labels, eval units, eval labels, parent SFT unit ids)."""
    from regslm.train.data import check_no_gold, fixture_split, load_train_units

    labels = read_label_rows(pcfg.labels)
    if pcfg.splits is None:  # fixture mode
        tu, tl, held = fixture_split(labels, pcfg.gold_sample, pcfg.eval_n)
        ev_l = {r.unit_id: r for r in held}
        ev_u = {r.unit_id: r.unit for r in held if r.unit is not None}
    else:
        import polars as pl

        tu = load_train_units(pcfg.splits)
        tl = labels
        sv = pl.read_parquet(pcfg.splits).filter(pl.col("split") == "silver_val")["unit_id"]
        ids = set(sv.to_list())
        ev_u = _units_by_id(ids)
        ev_l = {u: r for u, r in labels.items() if u in ids}
    ex = pcfg.run_dir(runs) / "train_examples.jsonl"
    sft_ids = (
        {json.loads(ln)["unit_id"] for ln in ex.read_text().splitlines() if ln.strip()}
        if ex.exists()
        else set()
    )
    check_no_gold(list(tu), pcfg.gold_sample)
    return tu, tl, ev_u, ev_l, sft_ids


def _units_by_id(ids: set[str]) -> dict[str, Unit]:
    import polars as pl

    frames = [pl.read_parquet("data/corpus/units.parquet")]
    local = Path("data/local/corpus/units_local.parquet")
    if local.exists():
        frames.append(pl.read_parquet(local))
    df = pl.concat(frames, how="diagonal").unique("unit_id").filter(pl.col("unit_id").is_in(ids))
    fields = list(Unit.model_fields)
    return {r["unit_id"]: Unit.model_validate(r) for r in df.select(fields).iter_rows(named=True)}


@dataclass
class _EvalSet:
    pairs: list[tuple[Unit, Extraction]]


def _eval_pairs(units: Mapping[str, Unit], rows: Mapping[str, LabelRow], n: int) -> _EvalSet:
    ids = [
        u for u in hash_order(rows)
        if u in units and rows[u].valid and rows[u].extraction is not None
    ]  # fmt: skip
    out: list[tuple[Unit, Extraction]] = []
    for u in ids[:n]:
        ex = rows[u].extraction
        assert ex is not None
        out.append((units[u], ex))
    return _EvalSet(out)


def greedy_texts(
    model: Any, tok: Any, units: Sequence[Unit], max_new: int, bs: int = 8
) -> list[str]:
    import torch

    from regslm.train.data import render_prompt

    was_training = bool(model.training)
    model.eval()
    side, tok.padding_side = tok.padding_side, "left"
    out: list[str] = []
    try:
        with torch.no_grad():
            for i in range(0, len(units), bs):
                prompts = [render_prompt(tok, student_messages(u)) for u in units[i : i + bs]]
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                gen = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                    pad_token_id=tok.pad_token_id,
                )  # fmt: skip
                new = gen[:, enc["input_ids"].shape[1] :]
                out += tok.batch_decode(new, skip_special_tokens=True)
    finally:
        tok.padding_side = side
        if was_training:
            model.train()
    return out


def _train(cfg: GrpoConfig, run_dir: Path, runs: Path = RUNS_DIR) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t0 = time.time()
    pcfg = TrainConfig.load(cfg.parent, runs)
    model, tok, backend, merge_base = _load_model(cfg, pcfg, runs)
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer

    from regslm.train.sft import merge_adapter

    set_seed(cfg.seed)
    tu, tl, ev_u, ev_l, sft_ids = _pool(pcfg, runs)
    ids, report = sample_prompts(tl, tu, sft_ids, cfg.n_prompts, cfg.numeric_frac, cfg.seed)
    if not ids:
        raise RuntimeError(f"no GRPO prompts: {report}")
    rows = prompt_rows(ids, tu, tl)
    (run_dir / "grpo_prompts.jsonl").write_text(
        "".join(json.dumps({"unit_id": u}) + "\n" for u in ids), encoding="utf-8"
    )
    ds = Dataset.from_list(rows)
    weights = load_weights(cfg.reward_weights)
    ev = _eval_pairs(ev_u, ev_l, cfg.eval_n)

    def eval_fn(m: Any, step: int) -> dict[str, float]:
        texts = greedy_texts(m, tok, [u for u, _ in ev.pairs], cfg.max_completion_length)
        return {"step": step, **eval_summary(texts, ev.pairs, weights)}

    log_path = run_dir / "reward_components.jsonl"
    log_path.write_text("")
    G = cfg.num_generations
    kw: dict[str, Any] = {}
    if cfg.loss_type is not None:
        kw["loss_type"] = cfg.loss_type
    if cfg.scale_rewards is not None:
        kw["scale_rewards"] = cfg.scale_rewards
    ckpt = run_dir / "checkpoints"
    args = GRPOConfig(
        output_dir=str(ckpt / "trainer"),
        per_device_train_batch_size=cfg.prompts_per_device * G,  # completions, not prompts
        gradient_accumulation_steps=cfg.prompts_per_step // cfg.prompts_per_device,
        num_generations=G,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=cfg.beta,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(cfg.warmup_frac * cfg.max_steps)),
        max_steps=cfg.max_steps,
        bf16=cfg.bf16,
        optim="adamw_torch",
        weight_decay=0.0,
        seed=cfg.seed,
        data_seed=cfg.seed,
        gradient_checkpointing=cfg.grad_ckpt,
        use_vllm=False,
        log_completions=True,
        chat_template_kwargs={"enable_thinking": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_only_model=True,
        report_to="none",
        **kw,
    )
    budget = BudgetCallback(cfg.max_wall_s)
    evcb = EvalCallback(cfg.eval_steps, eval_fn)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(weights, log_path),
        args=args,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[evcb, budget],  # type: ignore[list-item]
    )
    trainer.train()
    train_s = time.time() - t0
    hack = hacking_flags(evcb.results)
    (run_dir / "hacking.json").write_text(json.dumps(hack, indent=2))
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    out = ckpt / "adapter"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    steps_done = int(trainer.state.global_step)
    metrics: dict[str, Any] = {
        "status": "stopped_by_budget" if budget.stopped else "ok",
        "stopped_by_budget": budget.stopped,
        "trainer_backend": backend,
        "parent": cfg.parent,
        "parent_kind": "fresh_lora_on_merged" if pcfg.method == "full" else "continued_adapter",
        "steps": steps_done,
        "num_generations": G,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "scale_rewards": args.scale_rewards,
        "hacked": hack["hacked"],
        "hacking_reasons": hack["reasons"],
        "prompts": report,
        "train_wall_s": train_s,
        "peak_vram_allocated_bytes": peak,
        **snapshot_info(pcfg.base, pcfg.revision),
    }
    del trainer, model
    torch.cuda.empty_cache()
    shim = SimpleNamespace(
        base=merge_base, revision=None if merge_base != pcfg.base else pcfg.revision
    )
    merge_adapter(shim, out, ckpt / "merged")  # type: ignore[arg-type]
    metrics["wall_s"] = time.time() - t0
    return metrics


def run_grpo(
    cfg: GrpoConfig,
    runs: Path = RUNS_DIR,
    trainer: Callable[[GrpoConfig, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train and write ``metrics.json``. OOM is a result (``does_not_fit``), exit code 0."""
    run_dir = cfg.run_dir(runs)
    run_dir.mkdir(parents=True, exist_ok=True)
    fn = trainer or (lambda c, d: _train(c, d, runs))
    t0 = time.time()
    try:
        metrics = fn(cfg, run_dir)
    except Exception as e:
        if not is_oom(e):
            raise
        metrics = {
            "status": "does_not_fit",
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "parent": cfg.parent,
            "num_generations": cfg.num_generations,
            "wall_s": time.time() - t0,
        }
        if cfg.retry_g2 and cfg.num_generations > 2:
            sub = f"{cfg.run_id}-g2"
            write_run_config(
                sub, cfg.parent, {**cfg.model_dump(exclude={"run_id"}), "num_generations": 2,
                                  "retry_g2": False}, runs
            )  # fmt: skip
            metrics["retry_run"] = sub
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            run_grpo(GrpoConfig.load(sub, runs), runs, trainer)
            return metrics
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
