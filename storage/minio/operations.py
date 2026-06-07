import io
from datetime import timedelta
from pathlib import Path

import structlog
from fastapi import UploadFile

from storage.minio.client import ensure_bucket, get_minio_client

logger = structlog.get_logger(__name__)


async def upload_fileobj(file: UploadFile, object_key: str) -> str:
    """Stream an UploadFile to MinIO. Returns the object key."""
    ensure_bucket()
    from app.core.config import settings
    client = get_minio_client()
    data = await file.read()
    client.put_object(
        bucket_name=settings.minio_bucket,
        object_name=object_key,
        data=io.BytesIO(data),
        length=len(data),
        content_type=file.content_type or "application/octet-stream",
    )
    logger.info("minio.upload", key=object_key, size=len(data))
    return object_key


def upload_file(local_path: Path, object_key: str) -> str:
    """Upload a local file to MinIO. Used by workers."""
    ensure_bucket()
    from app.core.config import settings
    client = get_minio_client()
    client.fput_object(
        bucket_name=settings.minio_bucket,
        object_name=object_key,
        file_path=str(local_path),
    )
    return object_key


def download_file(object_key: str, dest_path: Path) -> Path:
    """Download an object from MinIO to a local path."""
    from app.core.config import settings
    client = get_minio_client()
    client.fget_object(
        bucket_name=settings.minio_bucket,
        object_name=object_key,
        file_path=str(dest_path),
    )
    return dest_path


def get_presigned_url(object_key: str, expires_hours: int = 1) -> str:
    """Return a presigned URL valid for `expires_hours` hours."""
    from app.core.config import settings
    client = get_minio_client()
    return client.presigned_get_object(
        bucket_name=settings.minio_bucket,
        object_name=object_key,
        expires=timedelta(hours=expires_hours),
    )


def delete_object(object_key: str) -> None:
    from app.core.config import settings
    client = get_minio_client()
    client.remove_object(settings.minio_bucket, object_key)
    logger.info("minio.delete", key=object_key)
