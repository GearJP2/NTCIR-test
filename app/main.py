from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints.search import router as search_router
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from storage.milvus.client import get_milvus_client
from storage.milvus.collections import ensure_all_collections

logger = structlog.get_logger(__name__)
CASTLE_MEDIA_DIR = Path("data/castle/videos")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    logger.info("startup", env=settings.app_env)

    # Ensure Milvus collections exist with correct schemas
    client = get_milvus_client()
    ensure_all_collections(client)
    logger.info("milvus.ready")

    yield

    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="CASTLE Hierarchical Multimodal Event Retrieval",
        version="0.1.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # Versioned API tree  →  /api/v1/...
    app.include_router(api_router, prefix="/api/v1")

    # Convenience alias   →  /api/search/episodic
    # Mirrors the v1 search router at a shorter path for direct client access.
    app.include_router(search_router, prefix="/api/search", tags=["search"])

    if CASTLE_MEDIA_DIR.exists():
        app.mount(
            "/media/castle",
            StaticFiles(directory=CASTLE_MEDIA_DIR),
            name="castle-media",
        )

    return app


app = create_app()
