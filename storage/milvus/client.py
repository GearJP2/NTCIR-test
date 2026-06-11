from functools import lru_cache

from pymilvus import MilvusClient


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    """Return a singleton MilvusClient. Called at app startup and reused across requests."""
    from app.core.config import settings
    return MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token or None,
        timeout=settings.milvus_timeout_sec,
    )
