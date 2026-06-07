from functools import lru_cache

from minio import Minio


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    from app.core.config import settings
    return Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket() -> None:
    from app.core.config import settings
    client = get_minio_client()
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)
