from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from services.events.manifest import load_event_manifest, write_event_manifest

app = typer.Typer(help="Merge per-recording CASTLE semantic outputs.")


@app.command()
def main(
    input_root: Path = typer.Argument(..., exists=True, file_okay=False),
    output_manifest: Path = typer.Option(
        Path("processed/lanta_semantic/semantic_events.jsonl")
    ),
    output_embeddings: Path = typer.Option(
        Path("processed/lanta_semantic/semantic_embeddings.npz")
    ),
) -> None:
    manifest_paths = sorted((input_root / "manifests").glob("*.jsonl"))
    embedding_paths = sorted((input_root / "embeddings").glob("*.npz"))
    records = []
    for path in manifest_paths:
        records.extend(load_event_manifest(path))
    write_event_manifest(output_manifest, records)

    event_ids = []
    embeddings = []
    for path in embedding_paths:
        payload = np.load(path)
        event_ids.extend(payload["event_ids"].tolist())
        embeddings.append(payload["embeddings"])
    output_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_embeddings,
        event_ids=np.array(event_ids),
        embeddings=np.concatenate(embeddings) if embeddings else np.empty((0, 0)),
    )
    summary = {
        "manifest_files": len(manifest_paths),
        "embedding_files": len(embedding_paths),
        "records": len(records),
        "embeddings": len(event_ids),
    }
    (output_manifest.parent / "merge_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(summary))


if __name__ == "__main__":
    app()
