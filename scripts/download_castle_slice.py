from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from services.dataset.castle_audit import CASTLE_REPO_ID

app = typer.Typer(help="Download selected CASTLE recordings with resumable transfers.")


@app.command()
def main(
    day: str = typer.Option(...),
    participant_id: str = typer.Option(...),
    recording_stem: list[str] = typer.Option(
        ...,
        help="Repeat for each recording stem to download.",
    ),
    revision: str = typer.Option(..., help="Pinned CASTLE repository revision."),
    output_dir: Path = typer.Option(
        Path("data/castle"),
        help="Local dataset root.",
    ),
) -> None:
    load_dotenv()
    rows: list[dict] = []
    for stem in recording_stem:
        repo_path = f"main/{day}/{participant_id}/video/{stem}.mp4"
        local_path = hf_hub_download(
            CASTLE_REPO_ID,
            repo_path,
            repo_type="dataset",
            revision=revision,
            token=True,
            local_dir=output_dir,
        )
        path = Path(local_path)
        rows.append(
            {
                "day": day,
                "participant_id": participant_id,
                "recording_stem": stem,
                "repo_path": repo_path,
                "local_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
            }
        )
        typer.echo(f"Ready: {repo_path} -> {path}")

    manifest_path = output_dir / "download_manifest.jsonl"
    existing = _load_existing(manifest_path)
    by_repo_path = {row["repo_path"]: row for row in [*existing, *rows]}
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False)
        for row in sorted(by_repo_path.values(), key=lambda row: row["repo_path"])
    )
    manifest_path.write_text(f"{payload}\n", encoding="utf-8")
    typer.echo(f"Updated {manifest_path}")


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    app()
