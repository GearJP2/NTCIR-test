"""Export saved video-reranker results; does not call models or remote services."""

import io
import json
from pathlib import Path

import typer
from pydantic import TypeAdapter

from evaluation.csat_submission import CSATResult, write_csat_submission


def main(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_path: Path = typer.Option(...),
    group_id: str = typer.Option("LLFL26"),
    run_id: str = typer.Option(...),
) -> None:
    if input_path.resolve() == output_path.resolve():
        raise typer.BadParameter("output must differ from input")
    results = TypeAdapter(dict[str, list[CSATResult]]).validate_python(
        json.loads(input_path.read_text(encoding="utf-8"))
    )
    buffer = io.StringIO()
    count = write_csat_submission(buffer, results, group_id=group_id, run_id=run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing runs, including when validation fails.
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(buffer.getvalue())
    typer.echo(f"Wrote {count} CSAT results to {output_path}")


if __name__ == "__main__":
    typer.run(main)
