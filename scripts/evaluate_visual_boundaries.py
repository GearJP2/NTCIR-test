from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import typer

from evaluation.boundary_metrics import evaluate_boundaries

app = typer.Typer(help="Evaluate CASTLE visual boundaries against a manual reference.")


@app.command()
def main(
    comparison_json: Path = typer.Argument(..., exists=True, dir_okay=False),
    reference_jsonl: Path = typer.Argument(..., exists=True, dir_okay=False),
    output_csv: Path = typer.Option(...),
) -> None:
    comparisons = json.loads(comparison_json.read_text(encoding="utf-8"))
    references = [
        json.loads(line)
        for line in reference_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference_ms = [int(row["timestamp_ms"]) for row in references]
    tolerances = {int(row["tolerance_ms"]) for row in references}
    if len(tolerances) != 1:
        raise typer.BadParameter("all manual references must use one tolerance")
    tolerance_ms = tolerances.pop()

    rows = []
    for comparison in comparisons:
        metrics = evaluate_boundaries(
            predicted_ms=comparison["boundary_timestamps_ms"],
            reference_ms=reference_ms,
            tolerance_ms=tolerance_ms,
        )
        rows.append(
            {
                "name": comparison["config"]["name"],
                "detector": comparison["config"]["detector"],
                "tolerance_ms": tolerance_ms,
                **asdict(metrics),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"Wrote {len(rows)} boundary evaluations to {output_csv}")


if __name__ == "__main__":
    app()
