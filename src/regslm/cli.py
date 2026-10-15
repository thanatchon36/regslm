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


if __name__ == "__main__":
    app()
