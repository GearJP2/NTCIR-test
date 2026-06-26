from __future__ import annotations

import json
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
    repository_files_jsonl: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional local JSONL with repository file rows for offline diagnostics.",
    ),
    revision: str = typer.Option(
        "c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4",
        help="Pinned CASTLE repository revision.",
    ),
) -> None:
    if repository_files_jsonl is None:
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
    else:
        files = _load_repository_files(repository_files_jsonl)
    rows = inspect_thermal_files(files)
    write_thermal_inventory(output_csv, rows)
    unassigned = sum(row.assignment_status == "unassigned" for row in rows)
    timestampless = sum(row.timestamp_status == "no_timestamp_in_path" for row in rows)
    typer.echo(
        f"Wrote {len(rows)} thermal rows to {output_csv}; "
        f"{unassigned} unassigned; {timestampless} without path timestamps."
    )


def _load_repository_files(path: Path) -> list[RepositoryFile]:
    files: list[RepositoryFile] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        files.append(
            RepositoryFile(
                path=row["path"],
                size_bytes=int(row.get("size_bytes") or 0),
            )
        )
    return files


if __name__ == "__main__":
    app()
