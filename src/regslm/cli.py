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


if __name__ == "__main__":
    app()
