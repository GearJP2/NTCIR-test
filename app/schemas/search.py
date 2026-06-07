from typing import Literal

from pydantic import BaseModel, Field


# ── Legacy multimodal search (audio_segments / visual / text collections) ────

class SearchRequest(BaseModel):
    text_query: str
    audio_url: str | None = None          # presigned MinIO URL or public URL
    top_k: int = Field(default=10, ge=1, le=100)
    modalities: list[Literal["audio", "visual", "text"]] = ["audio", "visual", "text"]


class RankedHit(BaseModel):
    segment_id: str
    media_id: str
    score: float
    start_sec: float
    end_sec: float
    transcript: str | None = None
    summary: str | None = None
    media_url: str | None = None


class SearchResponse(BaseModel):
    results: list[RankedHit]
    total: int


# ── Episodic memory search (csat_episodic_memory collection) ─────────────────

class EpisodicSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1024,
        description=(
            "Natural-language query describing the moment to retrieve. "
            "E.g. 'Find the moment where the user talked about a machine learning project'."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of nearest-neighbour chunks to retrieve from Milvus.",
    )
    media_id_filter: str | None = Field(
        default=None,
        description="Restrict the search to a single media file by its ID.",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score to include a hit (0–1).",
    )
    use_llm: bool = Field(
        default=True,
        description=(
            "When True, a WorldMM-style prompt is sent to an LLM to reason over "
            "the retrieved chunks and pinpoint the exact interaction event. "
            "Set to False to return raw vector hits only."
        ),
    )
    embedder: Literal["clap", "wav2vec2"] = Field(
        default="clap",
        description="Embedding model used at ingest time. Must match the collection.",
    )


class EpisodicHit(BaseModel):
    chunk_id: str
    media_id: str
    score: float                     # cosine similarity (0–1)
    start_sec: float
    end_sec: float
    duration_sec: float
    transcript: str
    language: str
    minio_url: str                   # presigned MinIO URL for the audio chunk
    object_key: str
    embedding_model: str
    created_at: int


class LLMReasoning(BaseModel):
    """Structured output produced by the WorldMM reasoning step."""
    answer: str                      # direct answer to the user's query
    best_chunk_ids: list[str]        # chunk IDs the LLM selected as most relevant
    reasoning: str                   # chain-of-thought explanation
    confidence: float                # self-reported confidence (0–1)
    model_used: str                  # e.g. "gpt-4o-mini" or "none"


class EpisodicSearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[EpisodicHit]
    reasoning: LLMReasoning | None   # None when use_llm=False or no LLM configured
    query_vector_dim: int
    embedder_used: str
