import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


class NotFoundError(Exception):
    pass


class StorageError(Exception):
    pass


class EmbeddingError(Exception):
    pass


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError):
        logger.error("storage.error", detail=str(exc))
        return JSONResponse(status_code=503, content={"detail": "Storage unavailable"})

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(request: Request, exc: EmbeddingError):
        logger.error("embedding.error", detail=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Embedding failed"})

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.exception("unhandled.error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
