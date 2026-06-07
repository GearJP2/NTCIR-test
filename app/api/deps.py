from storage.milvus.client import get_milvus_client
from storage.minio.client import get_minio_client


def get_db():
    """FastAPI dependency: yields the Milvus client."""
    return get_milvus_client()


def get_storage():
    """FastAPI dependency: yields the MinIO client."""
    return get_minio_client()
