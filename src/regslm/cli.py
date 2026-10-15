"""``regslm`` command-line interface (paper §6 "Library")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def doctor(out: Path = Path("results/hardware.json")) -> None:
    """Check CUDA, VRAM, WSL memory, disk and vLLM FP8 support; write results/hardware.json."""
    from regslm import doctor as d

    typer.echo(json.dumps(d.run(out), indent=2))


@app.command()
def schema(out: Path = Path("schemas/extraction.schema.json")) -> None:
    """Export the obligation JSON Schema."""
    from regslm.schema import json_schema

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_schema() + "\n")
    typer.echo(str(out))


@app.command()
def label(
    run_id: Annotated[str, typer.Option(help="Run directory name under results/runs/")],
    units: Annotated[Path, typer.Option(help="JSONL of corpus Unit rows")],
    variant: Annotated[str, typer.Option(help="label | rationale | gold_a | gold_b")] = "label",
    model: Annotated[str, typer.Option(help="HF model id")] = "Qwen/Qwen3-14B-AWQ",
    limit: Annotated[int | None, typer.Option(help="Label at most N units (after shuffle)")] = None,
    shuffle_seed: Annotated[int | None, typer.Option(help="Shuffle units with this seed")] = None,
    runs_dir: Path = Path("results/runs"),
) -> None:
    """Label corpus units with the local vLLM teacher (cached, resumable)."""
    import yaml

    from regslm.corpus.base import Unit
    from regslm.teacher.engine import EngineConfig
    from regslm.teacher.runner import label_units, shuffled

    if variant not in ("label", "rationale", "gold_a", "gold_b"):
        raise typer.BadParameter(f"unknown variant {variant!r}", param_hint="--variant")
    rows = [
        Unit.model_validate_json(line)
        for line in units.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cfg = EngineConfig(model=model)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "run_id": run_id,
        "units": str(units),
        "variant": variant,
        "limit": limit,
        "shuffle_seed": shuffle_seed,
        "engine": cfg.model_dump(),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    label_units(shuffled(rows, shuffle_seed), variant, cfg, run_dir, limit)  # type: ignore[arg-type]
    typer.echo(str(run_dir / "labels.jsonl"))


_SOURCES = Annotated[list[str] | None, typer.Argument(help="Source names (default: all tier-P)")]


@app.command()
def fetch(sources: _SOURCES = None) -> None:
    """Fetch raw documents for each source (cached, polite; re-runs are offline)."""
    from regslm.corpus import snapshot as sn
    from regslm.corpus.registry import ADAPTERS

    paths = sn.Paths()
    names = sn.resolve_sources(ADAPTERS, sources, sn.load_excluded(paths.config))
    for n, c in sn.fetch_sources(ADAPTERS, names, paths).items():
        typer.echo(f"{n}: {c} raw responses")


@app.command()
def parse(sources: _SOURCES = None) -> None:
    """Parse cached raw docs into data/local/{units,documents}/<source>.jsonl."""
    from regslm.corpus import snapshot as sn
    from regslm.corpus.registry import ADAPTERS

    paths = sn.Paths()
    names = sn.resolve_sources(ADAPTERS, sources, sn.load_excluded(paths.config))
    for n, (nd, nu) in sn.parse_sources(ADAPTERS, names, paths).items():
        typer.echo(f"{n}: {nd} documents, {nu} units")


@app.command()
def snapshot() -> None:
    """Build data/corpus/{manifest,documents,units}.parquet, VERSION and STATS.md."""
    from regslm.corpus import snapshot as sn
    from regslm.corpus.registry import ADAPTERS

    res = sn.build_snapshot(ADAPTERS, sn.Paths())
    typer.echo(
        f"{res.version}: {res.n_documents} docs, {res.n_units_public} public units, "
        f"{res.n_units_local} local units"
    )


@app.command()
def split(config: Path = Path("config/splits.yaml"), out: Path = Path("data/splits")) -> None:
    """Build data/splits/{assignments.parquet,SPLITS.md} (grouped, held-out, post-cutoff)."""
    from regslm.data.splits import build_splits, load_config

    a = build_splits(load_config(config), out=out)
    typer.echo(f"{a.height} units -> {out}")


@app.command("perturb-test")
def perturb_test(
    labels: Annotated[Path, typer.Option(help="Teacher labels.jsonl")],
    out: Path = Path("data/local/test_perturbed.jsonl"),
    splits: Path = Path("data/splits/assignments.parquet"),
    config: Path = Path("config/splits.yaml"),
    seed: int = 0,
) -> None:
    """Perturb numbers in split=test units that have a valid label; write test_perturbed.jsonl."""
    import polars as pl

    from regslm.corpus.base import Unit
    from regslm.data.perturb_set import build_perturbed_test, read_labels

    test_ids = set(pl.read_parquet(splits).filter(pl.col("split") == "test")["unit_id"].to_list())
    frames = [pl.read_parquet("data/corpus/units.parquet")]
    local = Path("data/local/corpus/units_local.parquet")
    if local.exists():
        frames.append(pl.read_parquet(local))
    rows = pl.concat(frames).filter(pl.col("unit_id").is_in(sorted(test_ids)))
    fields = list(Unit.model_fields)
    units = [Unit.model_validate(r) for r in rows.select(fields).iter_rows(named=True)]
    n = build_perturbed_test(units, read_labels(labels), out, seed)
    typer.echo(f"{n} perturbed test units -> {out}")


@app.command("eval-sets")
def eval_sets(
    labels: Annotated[Path, typer.Option(help="Teacher labels.jsonl (references)")],
    out: Path = Path("data/local/eval_sets"),
    splits: Path = Path("data/splits/assignments.parquet"),
    gold: Path = Path("data/gold/gold.jsonl"),
) -> None:
    """Materialise test/perturbed/mr_pool/... reference sets + MANIFEST.json (RQ doc section 0)."""
    from regslm.data.eval_sets import build_from_files

    m = build_from_files(labels, out, splits, gold)
    for name, v in m["sets"].items():
        typer.echo(f"{name}: n={v['n']} targets={v['n_targets']}")
    typer.echo(f"-> {out} (labels sha256 {m['labels_sha256'][:12]})")


gold_app = typer.Typer(no_args_is_help=True, add_completion=False, help="Gold set (paper 3.3).")
app.add_typer(gold_app, name="gold")


@gold_app.command("build")
def gold_build(
    run_a: Path = Path("results/runs/gold-pass-a"),
    run_b: Path = Path("results/runs/gold-pass-b"),
    sample: Path = Path("data/gold/sample.jsonl"),
    out: Path = Path("data/gold"),
) -> None:
    """Compare gold passes A and B; write agreement.json, to_adjudicate.jsonl, agreed.jsonl."""
    import polars as pl

    from regslm.eval import gold

    ids = [json.loads(ln)["unit_id"] for ln in sample.read_text().splitlines() if ln.strip()]
    frames = [pl.read_parquet("data/corpus/units.parquet")]
    local = Path("data/local/corpus/units_local.parquet")
    if local.exists():
        frames.append(pl.read_parquet(local))
    rows = pl.concat(frames, how="diagonal").filter(pl.col("unit_id").is_in(ids))
    units = {r["unit_id"]: r for r in rows.iter_rows(named=True)}
    summary = gold.build(run_a, run_b, sample, units, out)
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "agreement"}))


@gold_app.command("finalize")
def gold_finalize(out: Path = Path("data/gold")) -> None:
    """Merge agreed + adjudicated rows into gold.jsonl (refuses if any decision is null)."""
    from regslm.eval import gold

    try:
        n = gold.finalize(out)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{n} gold units -> {out / 'gold.jsonl'}")


@app.command("train-config")
def train_config(
    run_id: Annotated[str, typer.Option(help="Run directory name under results/runs/")],
    base: str = "Qwen/Qwen3-0.6B",
    method: str = "lora",
    n_labels: int = 500,
    arm: str = "label",
    seed: int = 0,
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="Extra KEY=VALUE (YAML value), repeatable")
    ] = None,
    runs_dir: Path = Path("results/runs"),
) -> None:
    """Write results/runs/<id>/config.yaml (pre-registered defaults + overrides)."""
    import yaml

    from regslm.train.config import TrainConfig

    d: dict[str, object] = {
        "run_id": run_id, "base": base, "method": method, "n_labels": n_labels,
        "arm": arm, "seed": seed,
    }  # fmt: skip
    for kv in set_ or []:
        k, _, v = kv.partition("=")
        d[k] = yaml.safe_load(v)
    typer.echo(str(TrainConfig.model_validate(d).save(runs_dir)))


@app.command()
def train(
    run_id: Annotated[str, typer.Option(help="Run directory with config.yaml")],
    runs_dir: Path = Path("results/runs"),
) -> None:
    """SFT / distillation run from results/runs/<id>/config.yaml (.venv-train)."""
    from regslm.train.config import TrainConfig
    from regslm.train.sft import run_training

    m = run_training(TrainConfig.load(run_id, runs_dir), runs_dir)
    typer.echo(f"{run_id}: {m['status']}")


@app.command()
def predict(
    run_id: Annotated[str, typer.Option(help="Trained run id")],
    sets: Annotated[str, typer.Option(help="Comma-separated test sets")] = "test",
    limit: Annotated[int | None, typer.Option(help="At most N units per set")] = None,
    refs: Annotated[
        str, typer.Option(help="Glob(s) of reference labels.jsonl, comma-separated")
    ] = "results/runs/silver-*/labels.jsonl",
    gpu_memory_utilization: float | None = None,
    runs_dir: Path = Path("results/runs"),
) -> None:
    """Greedy, unguided vLLM inference on test sets -> predictions/<set>.jsonl (.venv)."""
    from regslm.train.predict import predict as run_predict

    meta = run_predict(
        run_id, sets.split(","), limit, refs.split(","), runs_dir, gpu_memory_utilization
    )
    typer.echo(json.dumps(meta["sets"]))


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="Run with predictions/")],
    sets: Annotated[str | None, typer.Option(help="Comma-separated subset of sets")] = None,
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples")] = 10_000,
    runs_dir: Path = Path("results/runs"),
) -> None:
    """Score predictions with regslm.eval; write results/runs/<id>/eval.json."""
    from regslm.train.evaluate import evaluate as run_eval

    res = run_eval(run_id, sets.split(",") if sets else None, n_boot, runs_dir)
    for name, r in res["sets"].items():
        m = r["metrics"]
        typer.echo(f"{name}: n={r['n_units']} OF1={m['of1']:.3f} valid={m['validity_rate']:.3f}")


@app.command("grpo-config")
def grpo_config(
    run_id: Annotated[str, typer.Option(help="New GRPO run id")],
    parent: Annotated[str | None, typer.Option(help="Parent SFT run id")] = None,
    auto_base: Annotated[
        str | None, typer.Option(help="Pick the best SFT run (silver_val OF1) for this base")
    ] = None,
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="Extra KEY=VALUE (YAML value), repeatable")
    ] = None,
    runs_dir: Path = Path("results/runs"),
) -> None:
    """Write config.yaml + grpo.yaml for a GRPO run (pre-registered defaults + overrides)."""
    import yaml

    from regslm.train.grpo import select_parent, write_run_config

    if (parent is None) == (auto_base is None):
        raise typer.BadParameter("give exactly one of --parent / --auto-base")
    pid = parent or select_parent(str(auto_base), runs_dir)
    ov: dict[str, object] = {}
    for kv in set_ or []:
        k, _, v = kv.partition("=")
        ov[k] = yaml.safe_load(v)
    write_run_config(run_id, pid, ov, runs_dir)
    typer.echo(f"{run_id}: parent={pid}")


@app.command()
def grpo(
    run_id: Annotated[str, typer.Option(help="Run directory with config.yaml + grpo.yaml")],
    runs_dir: Path = Path("results/runs"),
) -> None:
    """GRPO with verifiable rewards from results/runs/<id>/grpo.yaml (.venv-train)."""
    from regslm.train.grpo import GrpoConfig, run_grpo

    m = run_grpo(GrpoConfig.load(run_id, runs_dir), runs_dir)
    typer.echo(f"{run_id}: {m['status']}")


@app.command()
def quantize(
    run_id: Annotated[str, typer.Option(help="Trained run with checkpoints/merged")],
    formats: Annotated[str, typer.Option(help="fp8,awq,gptq,gguf-q8,gguf-q4km")] = (
        "fp8,awq,gptq,gguf-q8,gguf-q4km"
    ),
    runs_dir: Path = Path("results/runs"),
) -> None:
    """Quantised exports of the merged checkpoint -> quant/<fmt>/, quant.json (.venv-quant)."""
    from regslm.quant.export import parse_formats
    from regslm.quant.export import quantize as run_quantize

    rec = run_quantize(run_id, parse_formats(formats), runs_dir)
    for f, r in rec["formats"].items():
        typer.echo(f"{f}: {r.get('status')} {r.get('size_bytes', '-')} B {r.get('error', '')}")


@app.command()
def bench(
    run_id: Annotated[str, typer.Option(help="Trained run (bf16 merged + quant/ exports)")],
    backend: Annotated[str, typer.Option(help="vllm | llamacpp-cuda | llamacpp-cpu")],
    format_: Annotated[
        str, typer.Option("--format", help="bf16|fp8-online|fp8|awq|gptq|gguf-q8|gguf-q4km")
    ],
    sets: str = "test",
    limit: Annotated[int | None, typer.Option(help="At most N units")] = None,
    threads: Annotated[int, typer.Option(help="llama.cpp CPU threads (6 P-cores / 14 all)")] = 6,
    concurrency: Annotated[int | None, typer.Option(help="Throughput-pass concurrency")] = None,
    spec_draft: Annotated[str | None, typer.Option(help="vLLM draft-model run id")] = None,
    spec_k: int = 3,
    needle: Annotated[bool, typer.Option(help="Also run needle-in-rulebook")] = False,
    needle_ctx: int = 17408,
    n_boot: int = 1000,
    runs_dir: Path = Path("results/runs"),
) -> None:
    """Throughput/latency/memory benchmark + eval -> results/bench/<run>/<backend>-<format>.json."""
    from regslm.bench.runner import bench as run_bench

    r = run_bench(
        run_id, backend, format_, sets=sets, limit=limit, threads=threads,
        concurrency=concurrency, spec_draft_run=spec_draft, spec_k=spec_k, runs=runs_dir,
        n_boot=n_boot, needle=needle, needle_ctx=needle_ctx,
    )  # fmt: skip
    typer.echo(json.dumps({k: r.get(k) for k in ("throughput", "single_stream", "memory")}))


if __name__ == "__main__":
    app()
