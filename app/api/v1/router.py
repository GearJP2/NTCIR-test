from app.api.v1.endpoints import health, ingest, query_expansion, search
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(query_expansion.router, prefix="/expand", tags=["query"])
