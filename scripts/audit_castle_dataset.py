from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

from services.dataset.castle_audit import (
    CASTLE_REPO_ID,
    RepositoryFile,
    build_castle_audit,
    write_audit_outputs,
)

app = typer.Typer(help="Audit CASTLE2024 repository structure without downloading media.")


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("processed/audit"),
        help="Directory for inventory, coverage, alignment, and report outputs.",
    ),
    revision: str = typer.Option("main", help="Hugging Face dataset revision."),
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
    audit = build_castle_audit(files)
    audit["revision"] = info.sha
    write_audit_outputs(output_dir, audit)
    typer.echo(
        f"Audited {audit['total_files']:,} CASTLE files "
        f"({audit['total_bytes'] / 1024**4:.2f} TiB); outputs: {output_dir}"
    )


if __name__ == "__main__":
    app()
