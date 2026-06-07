"""
POST /ingest/audio         — sync pipeline (dev / small files ≤ ~50 MB)
POST /ingest/audio/async   — background worker via ARQ (large files)
GET  /ingest/audio/{media_id}/chunks — list indexed chunks for a media file
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Literal

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.ingest import IngestResponse
from app.schemas.media import AudioIngestResponse, ChunkResult
from services.audio_service import AudioService, ChunkingStrategy

logger = structlog.get_logger(__name__)
router = APIRouter()

_ALLOWED_CONTENT_TYPES = frozenset({
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/flac", "audio/aac", "audio/m4a",
    "video/mp4", "video/webm", "video/x-matroska", "video/quicktime",
})

_VIDEO_TYPES = frozenset({
    "video/mp4", "video/webm", "video/x-matroska", "video/quicktime",
})


# ── Sync endpoint (small files, dev / demo) ───────────────────────────────────

@router.post("/audio", response_model=AudioIngestResponse, status_code=200)
async def ingest_audio_sync(
    file: UploadFile = File(..., description="Audio or video file to index"),
    media_id: str = Form(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID for this media. Re-using the same ID overwrites existing chunks.",
    ),
    language: str = Form(default="th", description="Primary spoken language (ISO 639-1)"),
    strategy: Literal["vad", "fixed_duration"] = Form(
        default="vad",
        description="'vad' uses speech boundaries; 'fixed_duration' splits at equal intervals.",
    ),
    embedder: Literal["clap", "wav2vec2"] = Form(
        default="clap",
        description="Embedding model: 'clap' → 512-dim, 'wav2vec2' → 768-dim.",
    ),
    chunk_duration_sec: float = Form(
        default=30.0,
        description="Target chunk length (seconds) — only for 'fixed_duration' strategy.",
        ge=5.0, le=300.0,
    ),
):
    """
    Full synchronous audio ingestion pipeline.

    Runs inline: segment → embed → upload chunks to MinIO →
    upsert to Milvus `csat_episodic_memory`.

    Returns the list of created chunks with their MinIO URLs.
    Use `/ingest/audio/async` for files larger than ~50 MB.
    """
    _validate_content_type(file.content_type or "")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = Path(file.filename or "upload").suffix or _content_type_to_suffix(file.content_type)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{media_id}{suffix}"
        tmp_path.write_bytes(raw_bytes)

        # Extract audio from video containers before processing
        if (file.content_type or "").split(";")[0].strip().lower() in _VIDEO_TYPES:
            from services.ingestion.video_processor import extract_audio_track
            tmp_path = extract_audio_track(tmp_path)

        # Archive raw source to MinIO (non-fatal if it fails)
        try:
            from storage.minio.operations import upload_file
            upload_file(tmp_path, f"raw/{media_id}/{file.filename or 'upload'}")
        except Exception as exc:
            logger.warning("ingest.raw_archive_failed", media_id=media_id, error=str(exc))

        service = AudioService(
            strategy=ChunkingStrategy(strategy),
            embedder_name=embedder,
            chunk_duration_sec=chunk_duration_sec,
        )

        try:
            chunks = service.run_pipeline(
                audio_path=tmp_path,
                media_id=media_id,
                language=language,
                tmp_dir=Path(tmp_dir),
            )
        except Exception as exc:
            logger.exception("ingest.pipeline_failed", media_id=media_id)
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion pipeline failed: {exc}",
            ) from exc

    logger.info(
        "ingest.audio.sync.done",
        media_id=media_id,
        total_chunks=len(chunks),
        strategy=strategy,
        embedder=embedder,
    )

    return AudioIngestResponse(
        media_id=media_id,
        total_chunks=len(chunks),
        strategy=strategy,
        embedder=embedder,
        chunks=[
            ChunkResult(
                chunk_id=c.chunk_id,
                media_id=c.media_id,
                start_sec=c.start_sec,
                end_sec=c.end_sec,
                duration_sec=c.duration_sec,
                minio_url=c.minio_url,
                object_key=c.object_key,
                language=c.language,
                embedding_model=c.embedding_model,
                created_at=c.created_at,
            )
            for c in chunks
        ],
    )


# ── Async endpoint (large files → ARQ worker) ─────────────────────────────────

@router.post("/audio/async", response_model=IngestResponse, status_code=202)
async def ingest_audio_async(
    file: UploadFile = File(...),
    media_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
    language: str = Form(default="th"),
    strategy: Literal["vad", "fixed_duration"] = Form(default="vad"),
    embedder: Literal["clap", "wav2vec2"] = Form(default="clap"),
):
    """
    Async ingestion via ARQ background worker.
    Uploads the raw file to MinIO, enqueues a job, and returns a `job_id`.
    Poll `/health/ready` or your job-status endpoint for completion.
    """
    _validate_content_type(file.content_type or "")

    from storage.minio.operations import upload_fileobj
    object_key = f"raw/{media_id}/{file.filename or 'upload'}"
    await upload_fileobj(file, object_key)

    from workers.tasks.ingest_task import enqueue_ingest
    job_id = await enqueue_ingest(
        media_id=media_id,
        object_key=object_key,
        title=file.filename or "",
        language=language,
        content_type=file.content_type or "audio/mpeg",
    )

    logger.info("ingest.audio.async.queued", media_id=media_id, job_id=job_id)
    return IngestResponse(media_id=media_id, job_id=job_id, status="queued")


# ── Chunk listing ─────────────────────────────────────────────────────────────

@router.get("/audio/{media_id}/chunks")
async def list_chunks(media_id: str, limit: int = 500):
    """Return all indexed chunks for a media file, sorted by start_sec."""
    from storage.milvus.milvus_service import MilvusService
    try:
        records = MilvusService().list_by_media_id(media_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"media_id": media_id, "total": len(records), "chunks": records}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_content_type(raw_content_type: str) -> None:
    base = raw_content_type.split(";")[0].strip().lower()
    if base not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type: '{raw_content_type}'. "
                f"Accepted: {sorted(_ALLOWED_CONTENT_TYPES)}"
            ),
        )


def _content_type_to_suffix(content_type: str | None) -> str:
    _MAP = {
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
        "audio/wav": ".wav",  "audio/x-wav": ".wav",
        "audio/flac": ".flac","audio/ogg": ".ogg",
        "audio/aac": ".aac",  "audio/m4a": ".m4a",
        "video/mp4": ".mp4",  "video/webm": ".webm",
        "video/x-matroska": ".mkv", "video/quicktime": ".mov",
    }
    base = (content_type or "").split(";")[0].strip().lower()
    return _MAP.get(base, ".bin")
