from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from services.dataset.castle_audit import CASTLE_REPO_ID
from services.dataset.castle_metadata import (
    inspect_gaze_csv,
    inspect_heart_rate_csv,
    inspect_main_metadata_csv,
    write_timeline_inventory,
)

app = typer.Typer(
    help="Inspect CASTLE metadata timestamp sources for a development slice."
)


@app.command()
def main(
    day: str = typer.Option("day1"),
    participant_id: str = typer.Option("Allie"),
    recording_stem: list[str] = typer.Option(["08", "09", "10"]),
    metadata_sensor: str = typer.Option("ACCL"),
    output_csv: Path = typer.Option(
        Path("processed/timeline/day1_Allie/source_timeline_inventory.csv")
    ),
    output_json: Path = typer.Option(
        Path("processed/timeline/day1_Allie/source_timeline_inventory.json")
    ),
    revision: str = typer.Option(
        "c8e7b5cd9e9c83d0ff42560fc1169bed7867abd4"
    ),
) -> None:
    load_dotenv(".env")
    inspections = []
    for stem in recording_stem:
        repo_path = (
            f"main/{day}/{participant_id}/metadata/{stem}.{metadata_sensor}.csv"
        )
        local_path = _download(repo_path, revision)
        inspections.append(
            inspect_main_metadata_csv(
                local_path,
                source=f"{day}/{participant_id}/{stem}.{metadata_sensor}",
            )
        )

    heart_path = _download(
        f"auxiliary/heartrate/{participant_id}/{day}.csv",
        revision,
    )
    inspections.append(
        inspect_heart_rate_csv(
            heart_path,
            source=f"auxiliary/heartrate/{participant_id}/{day}",
        )
    )

    gaze_path = _download(f"auxiliary/gaze/{participant_id}.csv", revision)
    inspections.append(
        inspect_gaze_csv(gaze_path, source=f"auxiliary/gaze/{participant_id}")
    )

    write_timeline_inventory(output_csv, inspections)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps([asdict(inspection) for inspection in inspections], indent=2)
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote {len(inspections)} timeline source rows to {output_csv}")


def _download(repo_path: str, revision: str) -> Path:
    return Path(
        hf_hub_download(
            CASTLE_REPO_ID,
            repo_path,
            repo_type="dataset",
            revision=revision,
            token=True,
        )
    )


if __name__ == "__main__":
    app()
