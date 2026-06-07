"""
CLI: Bulk-ingest NTCIR media corpus from a local directory.
Usage: python scripts/ingest_corpus.py --corpus-dir data/sample/ --language th
"""

import asyncio
from pathlib import Path

import typer
from tqdm import tqdm

app = typer.Typer()

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".ogg"}


@app.command()
def main(
    corpus_dir: Path = typer.Option(..., help="Directory containing media files"),
    language: str = typer.Option("th", help="Primary language for ASR"),
    workers: int = typer.Option(2, help="Number of concurrent ingestion workers"),
):
    files = [f for f in corpus_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    typer.echo(f"Found {len(files)} media files in {corpus_dir}")
    asyncio.run(_ingest_all(files, language, workers))


async def _ingest_all(files: list[Path], language: str, max_concurrency: int):
    import uuid
    from app.schemas.media import MediaAsset
    from services.ingestion.pipeline import run_ingestion_pipeline

    sem = asyncio.Semaphore(max_concurrency)

    async def _ingest(path: Path):
        async with sem:
            ext = path.suffix.lower()
            content_type = "video/mp4" if ext in {".mp4", ".mkv", ".webm"} else "audio/mpeg"
            asset = MediaAsset(
                media_id=str(uuid.uuid4()),
                object_key=f"raw/{path.name}",
                content_type=content_type,
                title=path.stem,
                language=language,
            )
            await run_ingestion_pipeline(asset, path)

    tasks = [_ingest(f) for f in files]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await coro


if __name__ == "__main__":
    app()
