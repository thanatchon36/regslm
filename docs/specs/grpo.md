# Spec: `regslm.train.grpo` — GRPO with verifiable rewards (paper §4.4, RQ3)

Environment `.venv-train` (TRL `GRPOTrainer`, via Unsloth when it supports the base; else
plain TRL + PEFT). Generation inside the trainer uses HF `generate` (no colocated vLLM: VRAM).

## Starting point and data
- Start from the **best SFT checkpoint per size** (0.6B, 1.7B, 4B), "best" = highest OF1 on
  `silver_val` (never test/gold) among that size's SFT runs; ties → smaller N. Continue training
  the same LoRA adapter (or a fresh LoRA on the merged SFT model for full-FT parents — record).
- Prompts: `n_prompts=2000` train units (teacher-labelled, valid, no unresolved) drawn from units
  **not** in the parent's SFT subset if available, else from it (record overlap fraction);
  20 % drawn from units whose silver label contains a threshold or deadline (numeric focus of
  H3a); same student prompt as SFT.
- References: silver `Extraction`s (resolved). Placebo GRPO is not run (budget); the SFT placebo
  covers the training-signal check.

## Reward
`regslm.rewards.composite` with `config/rewards.yaml` as frozen at `prereg-v1`. The reward
function receives completions + the batch's units and references (pass via dataset columns
`unit_json`, `reference_json`). Per-component means are logged every step (TRL
`log_completions` + custom callback) to `results/runs/<id>/reward_components.jsonl`.

## Hyperparameters (pre-registered defaults)
`num_generations=8` (0.6B/1.7B), 4 (4B); `max_completion_length=512`; temperature 1.0, top_p 1.0;
`beta` (KL) = 0.04; LoRA lr 5e-6, cosine, warmup 5 %; per-device prompts 2 with grad accumulation
to 16 prompts/step; `max_steps=250` (= 4,000 prompt visits, 2 epochs of 2,000); `loss_type`
= TRL default for the installed version (record it); `scale_rewards` TRL default (record).
Checkpoint every 50 steps; evaluate on `silver_val` (greedy, 300 units) at 0/100/250 steps.

## Reward-hacking guards (RQ3; all logged, none silently "fixed")
- Per step: fraction of empty extractions, mean completion length, mean #obligations,
  per-component reward, recall term.
- Flag `hacked=True` in `metrics.json` if, at any eval point vs step 0: empty-rate +10 pp, or
  recall −5 pp while total reward rises (RESEARCH_QUESTIONS RQ3 rule). A flagged run is still
  reported; the final checkpoint is still the step-250 one (no cherry-picking by test metrics).
- Length guard and recall term live in the reward (rewards spec), not in the trainer.

## Budget and OOM
Per-run wall-time cap 3 h (`max_wall_s`, checked in a callback → save + stop + record
`stopped_by_budget`). OOM → `status: does_not_fit` result as in SFT. 4B with 4 generations ×
512 tokens and gradient checkpointing is expected to fit in ~10 GB; if not, record and retry
once with `num_generations=2` as a separate, labelled cell.

## Outputs
Same run registry as SFT, plus `reward_components.jsonl`, `hacking.json`. Then the standard
`regslm predict` + `regslm evaluate` on all test sets.

## Tests (CPU)
Reward wrapper: maps TRL's (prompts, completions, **kwargs) to `composite` correctly incl.
batch alignment of `unit_json`/`reference_json`; hacking-flag logic on synthetic logs; budget
callback stops; prompt sampler (numeric share, overlap accounting).
