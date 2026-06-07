from pydantic import BaseModel


class AudioSegment(BaseModel):
    segment_id: str
    media_id: str
    start_sec: float
    end_sec: float
    audio_path: str          # local temp path


# ── Ingestion API models ──────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    """Serialisable summary of one indexed audio chunk."""
    chunk_id: str
    media_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    minio_url: str
    object_key: str
    language: str
    embedding_model: str
    created_at: int


class AudioIngestResponse(BaseModel):
    """Returned by POST /ingest/audio after a successful pipeline run."""
    media_id: str
    total_chunks: int
    chunks: list[ChunkResult]
    strategy: str
    embedder: str


class VideoKeyframe(BaseModel):
    frame_id: str
    media_id: str
    timestamp_sec: float
    image_path: str          # local temp path


class TranscriptChunk(BaseModel):
    chunk_id: str
    media_id: str
    segment_id: str
    text: str
    start_sec: float
    end_sec: float
    language: str


class MediaAsset(BaseModel):
    media_id: str
    object_key: str          # MinIO key
    content_type: str
    title: str = ""
    language: str = "th"
    duration_sec: float | None = None
