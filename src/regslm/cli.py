"""``regslm`` command-line interface (paper §6 "Library")."""

from __future__ import annotations

import json
from pathlib import Path

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


if __name__ == "__main__":
    app()
