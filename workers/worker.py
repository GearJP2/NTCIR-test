from arq.connections import RedisSettings

from app.core.config import settings
from workers.tasks.ingest_task import ingest_job


async def startup(ctx: dict):
    from model_zoo.registry import ModelRegistry
    ModelRegistry.preload_all()


async def shutdown(ctx: dict):
    pass


class WorkerSettings:
    functions = [ingest_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 4
    job_timeout = 3600          # 1 hour max per ingestion job
    keep_result = 86400         # keep results for 24h
