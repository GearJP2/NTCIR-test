import uuid
from pathlib import Path
import tempfile

import structlog

from app.schemas.media import MediaAsset
from services.ingestion.pipeline import run_ingestion_pipeline
from storage.minio.operations import download_file

logger = structlog.get_logger(__name__)


async def ingest_job(
    ctx: dict,
    *,
    media_id: str,
    object_key: str,
    title: str,
    language: str,
    content_type: str,
) -> dict:
    """
    ARQ task: download media from MinIO → run full ingestion pipeline.
    `ctx` is injected by ARQ and contains the Redis connection pool.
    """
    logger.info("worker.ingest.start", media_id=media_id, key=object_key)

    asset = MediaAsset(
        media_id=media_id,
        object_key=object_key,
        content_type=content_type,
        title=title,
        language=language,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        suffix = Path(object_key).suffix or ".mp4"
        local_path = Path(tmp_dir) / f"{media_id}{suffix}"
        download_file(object_key, local_path)
        result = await run_ingestion_pipeline(asset, local_path)

    logger.info("worker.ingest.done", **result)
    return result


async def enqueue_ingest(**kwargs) -> str:
    """Enqueue an ingest_job and return the ARQ job ID."""
    from arq import create_pool
    from app.core.config import settings
    pool = await create_pool({"host": settings.redis_url.replace("redis://", "").split("/")[0]})
    job = await pool.enqueue_job("ingest_job", **kwargs)
    await pool.close()
    return job.job_id
