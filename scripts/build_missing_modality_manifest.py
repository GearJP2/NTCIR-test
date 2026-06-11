from __future__ import annotations

import json
from pathlib import Path

import typer

from evaluation.manifest import load_evaluation_manifest
from storage.milvus.client import get_milvus_client
from storage.milvus.collections import ensure_all_collections
from storage.milvus.schemas import AUDIO_COLLECTION, TEXT_COLLECTION, VISUAL_COLLECTION

app = typer.Typer(help="Build a manifest containing media missing selected modalities.")

MODALITY_COLLECTIONS = {
    "visual": VISUAL_COLLECTION,
    "audio": AUDIO_COLLECTION,
    "asr": TEXT_COLLECTION,
}


@app.command()
def main(
    manifest_path: Path = typer.Option(..., help="Source evaluation manifest JSONL."),
    output_path: Path = typer.Option(..., help="Output manifest JSONL."),
    modality: list[str] = typer.Option(
        ...,
        "--modality",
        help="Required modality. Repeat for multiple modalities.",
    ),
) -> None:
    selected = {item.lower() for item in modality}
    unknown = selected.difference(MODALITY_COLLECTIONS)
    if unknown:
        raise typer.BadParameter(
            f"modality must be one of: {', '.join(sorted(MODALITY_COLLECTIONS))}"
        )

    videos = load_evaluation_manifest(manifest_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_media_id = {row["media_id"]: row for row in rows}

    client = get_milvus_client()
    ensure_all_collections(client)

    missing_media_ids: list[str] = []
    complete = 0
    partial = 0
    for video in videos:
        readiness = {
            item: _has_modality(client, video.media_id, MODALITY_COLLECTIONS[item])
            for item in selected
        }
        if all(readiness.values()):
            complete += 1
            continue
        if any(readiness.values()):
            partial += 1
        missing_media_ids.append(video.media_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(rows_by_media_id[media_id], ensure_ascii=False) + "\n"
            for media_id in missing_media_ids
        ),
        encoding="utf-8",
    )

    typer.echo(f"required_modalities={','.join(sorted(selected))}")
    typer.echo(f"complete={complete}")
    typer.echo(f"partial={partial}")
    typer.echo(f"missing={len(missing_media_ids)}")
    typer.echo(f"wrote={output_path}")


def _has_modality(client, media_id: str, collection_name: str) -> bool:
    rows = client.query(
        collection_name=collection_name,
        filter=f'media_id == "{media_id}"',
        output_fields=["media_id"],
        limit=1,
    )
    return bool(rows)


if __name__ == "__main__":
    app()
