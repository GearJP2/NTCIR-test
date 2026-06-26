from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

from services.dataset.castle_audit import CASTLE_REPO_ID, RepositoryFile
from services.dataset.castle_thermal import inspect_thermal_files, write_thermal_inventory

app = typer.Typer(help="Inventory CASTLE thermal files without downloading images.")


@app.command()
def main(
    output_csv: Path = typer.Option(
        Path("processed/timeline/thermal_inventory.csv"),
        help="Output thermal provenance CSV.",
    ),
    revision: str = typer.Option(
        "c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4",
        help="Pinned CASTLE repository revision.",
    ),
) -> None:
    load_dotenv()
    info = HfApi().dataset_info(
        CASTLE_REPO_ID,
        revision=revision,
        token=True,
        files_metadata=True,
    )
    files = [
        RepositoryFile(
            path=file.rfilename,
            size_bytes=int(getattr(file, "size", 0) or 0),
        )
        for file in info.siblings
    ]
    rows = inspect_thermal_files(files)
    write_thermal_inventory(output_csv, rows)
    unassigned = sum(row.assignment_status == "unassigned" for row in rows)
    timestampless = sum(row.timestamp_status == "no_timestamp_in_path" for row in rows)
    typer.echo(
        f"Wrote {len(rows)} thermal rows to {output_csv}; "
        f"{unassigned} unassigned; {timestampless} without path timestamps."
    )


if __name__ == "__main__":
    app()
