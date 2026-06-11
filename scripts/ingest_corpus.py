"""
CLI: Bulk-ingest NTCIR media corpus from a local directory.
Usage: python scripts/ingest_corpus.py --corpus-dir data/sample/ --language th
"""

import asyncio
import uuid
from pathlib import Path

import typer
from tqdm import tqdm

app = typer.Typer()

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".ogg"}
MEDIA_ID_SOURCES = {"uuid", "filename"}
SUPPORTED_MODALITIES = {"visual", "audio", "asr"}
MODALITY_COLLECTIONS = {
    "visual": "visual_keyframes",
    "audio": "audio_segments",
    "asr": "text_transcripts",
}


def media_id_for_path(path: Path, source: str) -> str:
    if source == "uuid":
        return str(uuid.uuid4())
    if source == "filename":
        return path.stem
    raise ValueError(f"Unsupported media_id source: {source}")


def media_files_from_directory(corpus_dir: Path) -> list[Path]:
    return [
        f
        for f in corpus_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and f.parent.name != "keyframes"
        and not f.name.endswith(".audio.wav")
    ]


def media_files_from_manifest(manifest_path: Path) -> list[Path]:
    from evaluation.manifest import load_evaluation_manifest

    return [video.video_path for video in load_evaluation_manifest(manifest_path)]


def media_files_starting_at(files: list[Path], media_id: str | None) -> list[Path]:
    if media_id is None:
        return files
    for index, path in enumerate(files):
        if path.stem == media_id:
            return files[index:]
    raise ValueError(f"media_id not found in input files: {media_id}")


def media_files_matching_ids(files: list[Path], media_ids: list[str]) -> list[Path]:
    if not media_ids:
        return files

    requested = set(media_ids)
    matched = [path for path in files if path.stem in requested]
    missing = requested.difference(path.stem for path in matched)
    if missing:
        raise ValueError(f"media_id not found in input files: {', '.join(sorted(missing))}")
    return matched


def media_files_without_indexed_modalities(
    files: list[Path],
    media_id_source: str,
    modalities: set[str],
    milvus_client,
) -> list[Path]:
    if not files:
        return []

    remaining: list[Path] = []
    for path in files:
        media_id = media_id_for_path(path, media_id_source)
        if not has_indexed_modalities(media_id, modalities, milvus_client):
            remaining.append(path)
    return remaining


def has_indexed_modalities(
    media_id: str,
    modalities: set[str],
    milvus_client,
) -> bool:
    for modality in modalities:
        collection_name = MODALITY_COLLECTIONS[modality]
        rows = milvus_client.query(
            collection_name=collection_name,
            filter=f'media_id == "{media_id}"',
            output_fields=["media_id"],
            limit=1,
        )
        if not rows:
            return False
    return True


@app.command()
def main(
    corpus_dir: Path = typer.Option(
        Path("data/sample"),
        help="Directory containing media files.",
    ),
    manifest_path: Path | None = typer.Option(
        None,
        help="Evaluation Manifest JSONL. When set, ingest only listed video_path values.",
    ),
    language: str = typer.Option("th", help="Primary language for ASR"),
    workers: int = typer.Option(2, help="Number of concurrent ingestion workers"),
    media_id_source: str = typer.Option(
        "uuid",
        help="How to assign media_id: uuid or filename.",
    ),
    start_at_media_id: str | None = typer.Option(
        None,
        help="Resume ingestion from this filename/media_id when using filename IDs.",
    ),
    only_media_id: list[str] | None = typer.Option(
        None,
        help="Ingest only these filename/media_id values. Can be passed multiple times.",
    ),
    modalities: list[str] | None = typer.Option(
        None,
        "--modality",
        "--modalities",
        help=(
            "Evidence modalities to index: visual, audio, asr. "
            "Repeat to select multiple. Defaults to all."
        ),
    ),
    keyframe_interval_sec: float | None = typer.Option(
        None,
        help=(
            "Override visual keyframe sampling interval for this ingestion run. "
            "Defaults to configs/model_config.yaml ingestion.keyframe_interval_sec."
        ),
    ),
    skip_indexed: bool = typer.Option(
        False,
        help="Skip media IDs that already have rows for all selected modalities.",
    ),
):
    if media_id_source not in MEDIA_ID_SOURCES:
        raise typer.BadParameter(
            f"media_id_source must be one of: {', '.join(sorted(MEDIA_ID_SOURCES))}"
        )
    selected_modalities = normalize_modalities(modalities)
    files = (
        media_files_from_manifest(manifest_path)
        if manifest_path is not None
        else media_files_from_directory(corpus_dir)
    )
    try:
        files = media_files_starting_at(files, start_at_media_id)
        files = media_files_matching_ids(files, only_media_id or [])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    missing = [path for path in files if not path.exists()]
    if missing:
        raise typer.BadParameter(f"{len(missing)} manifest media files are missing")

    if skip_indexed:
        from storage.milvus.client import get_milvus_client
        from storage.milvus.collections import ensure_all_collections

        milvus = get_milvus_client()
        ensure_all_collections(milvus)
        before = len(files)
        files = media_files_without_indexed_modalities(
            files,
            media_id_source,
            selected_modalities,
            milvus,
        )
        typer.echo(f"Skipped {before - len(files)} already indexed media files")

    source = manifest_path if manifest_path is not None else corpus_dir
    typer.echo(f"Found {len(files)} media files from {source}")
    typer.echo(f"Indexing modalities: {', '.join(sorted(selected_modalities))}")
    if keyframe_interval_sec is not None:
        typer.echo(f"Keyframe interval override: {keyframe_interval_sec}s")
    asyncio.run(
        _ingest_all(
            files,
            language,
            workers,
            media_id_source,
            selected_modalities,
            keyframe_interval_sec,
        )
    )


def normalize_modalities(modalities: list[str] | None) -> set[str]:
    if not modalities:
        return set(SUPPORTED_MODALITIES)
    selected = {modality.lower() for modality in modalities}
    unknown = selected.difference(SUPPORTED_MODALITIES)
    if unknown:
        raise typer.BadParameter(
            f"modalities must be one or more of: {', '.join(sorted(SUPPORTED_MODALITIES))}"
        )
    return selected


async def _ingest_all(
    files: list[Path],
    language: str,
    max_concurrency: int,
    media_id_source: str,
    modalities: set[str],
    keyframe_interval_sec: float | None,
):
    from app.schemas.media import MediaAsset
    from services.ingestion.pipeline import run_ingestion_pipeline

    sem = asyncio.Semaphore(max_concurrency)

    async def _ingest(path: Path):
        async with sem:
            ext = path.suffix.lower()
            content_type = "video/mp4" if ext in {".mp4", ".mkv", ".webm"} else "audio/mpeg"
            media_id = media_id_for_path(path, media_id_source)
            asset = MediaAsset(
                media_id=media_id,
                object_key=f"raw/{path.name}",
                content_type=content_type,
                title=path.stem,
                language=language,
            )
            await run_ingestion_pipeline(
                asset,
                path,
                modalities=modalities,
                keyframe_interval_sec=keyframe_interval_sec,
            )

    tasks = [_ingest(f) for f in files]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await coro


if __name__ == "__main__":
    app()
