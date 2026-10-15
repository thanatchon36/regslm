"""Training run configuration (pydantic, YAML in ``results/runs/<id>/config.yaml``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Method = Literal["lora", "qlora", "full"]
Arm = Literal["label", "rationale", "placebo"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RUNS_DIR = Path("results/runs")


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    base: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    method: Method = "lora"
    n_labels: int = 500
    arm: Arm = "label"
    seed: int = 0
    lr: float | None = None  # None -> 2e-4 (lora/qlora) / 2e-5 (full)
    epochs: int = 2
    min_steps: int = 300
    max_steps: int | None = None  # hard override (quick test)
    per_device_batch: int = 4
    grad_accum: int = 4  # effective batch 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = Field(default_factory=lambda: list(LORA_TARGETS))
    max_seq_len: int = 2048
    optim: str | None = None  # None -> adamw_8bit for full, else adamw_torch
    bf16: bool = True
    grad_ckpt: bool = True
    eval_n: int = 0  # fixture mode: rows held out of training for evaluation
    backend: Literal["auto", "unsloth", "plain"] = "auto"
    # data
    labels: list[str] = Field(
        default_factory=lambda: ["results/runs/silver-train-label/labels.jsonl"]
    )
    rationale_labels: list[str] = Field(default_factory=list)
    splits: str | None = "data/splits/assignments.parquet"  # None: fixture mode, no split filter
    gold_sample: str = "data/gold/sample.jsonl"
    match_rationale_tokens: bool = False  # RQ2 equal-token-budget label arm
    rule_empty: bool = True
    langs: list[str] | None = None  # restrict train units to these languages (None: all, en+th)

    @property
    def effective_lr(self) -> float:
        return self.lr if self.lr is not None else (2e-5 if self.method == "full" else 2e-4)

    @property
    def effective_optim(self) -> str:
        return self.optim or ("adamw_8bit" if self.method == "full" else "adamw_torch")

    def hash(self) -> str:
        """Hash of everything but the run id (identical settings -> identical hash)."""
        d = self.model_dump(mode="json")
        d.pop("run_id")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def run_dir(self, runs: Path = RUNS_DIR) -> Path:
        return runs / self.run_id

    def save(self, runs: Path = RUNS_DIR) -> Path:
        p = self.run_dir(runs) / "config.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
        return p

    @classmethod
    def load(cls, run_id: str, runs: Path = RUNS_DIR) -> TrainConfig:
        d: dict[str, Any] = yaml.safe_load((runs / run_id / "config.yaml").read_text())
        return cls.model_validate(d)


def total_steps(n_examples: int, cfg: TrainConfig) -> int:
    """Optimizer steps: 2 epochs but at least ``min_steps`` (small N get more epochs)."""
    if cfg.max_steps is not None:
        return cfg.max_steps
    eff = cfg.per_device_batch * cfg.grad_accum
    per_epoch = max(1, -(-n_examples // eff))
    return max(cfg.epochs * per_epoch, cfg.min_steps)
