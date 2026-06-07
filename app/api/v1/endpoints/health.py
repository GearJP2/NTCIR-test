from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    env: str


@router.get("/health", response_model=HealthResponse)
async def liveness():
    from app.core.config import settings
    return HealthResponse(status="ok", env=settings.app_env)


@router.get("/ready", response_model=HealthResponse)
async def readiness():
    """Check Milvus and MinIO connectivity."""
    from app.core.config import settings
    from storage.milvus.client import get_milvus_client
    client = get_milvus_client()
    client.list_collections()  # raises if unreachable
    return HealthResponse(status="ready", env=settings.app_env)
