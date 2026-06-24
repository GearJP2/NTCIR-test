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


# ── Moment search (canonical long-video benchmark contract) ─────────────────

class Evidence(BaseModel):
    source_type: Literal["visual", "audio", "asr", "summary"]
    score: float = Field(ge=0.0)
    source_id: str | None = None
    timestamp_sec: float | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    text: str | None = None


class VideoMoment(BaseModel):
    rank: int = Field(ge=1)
    moment_id: str
    media_id: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    score: float = Field(ge=0.0)
    thumbnail_sec: float = Field(ge=0.0)
    evidence: list[Evidence] = Field(default_factory=list)


class MomentSearchRequest(BaseModel):
    media_id: str = Field(
        ...,
        min_length=1,
        description="Selected video/media ID that bounds the single-video search scope.",
    )
    query: str = Field(
        ...,
        min_length=3,
        max_length=1024,
        description="Natural-language Semantic Query describing the moment to retrieve.",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    duration_sec: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional selected-video duration used for fixed-window generation.",
    )
    window_sec: float = Field(
        default=10.0,
        gt=0.0,
        description="Fixed moment window duration in seconds.",
    )
    stride_sec: float = Field(
        default=5.0,
        gt=0.0,
        description="Fixed moment window stride in seconds.",
    )
    profile: str = Field(
        default="activitynet_visual_heavy",
        description="Evaluation Profile controlling modality weights and matching assumptions.",
    )


class CollectionMomentSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1024,
        description="Natural-language Semantic Query searched across an evaluation collection.",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    manifest_path: str = Field(
        default="data/manifests/activitynet_dev200_indexed_current.jsonl",
        description="ActivityNet-style manifest that defines the searchable collection.",
    )
    max_videos: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Optional cap for interactive debugging; null searches the full manifest.",
    )
    candidate_limit: int = Field(
        default=1000,
        ge=1,
        le=5000,
        description="Per-modality ANN hit count before temporal window aggregation.",
    )
    window_sec: float = Field(
        default=10.0,
        gt=0.0,
        description="Fixed moment window duration in seconds.",
    )
    stride_sec: float = Field(
        default=5.0,
        gt=0.0,
        description="Fixed moment window stride in seconds.",
    )
    profile: str = Field(
        default="activitynet_visual_only",
        description="Evaluation Profile controlling modality weights and matching assumptions.",
    )


class GroundTruthInterval(BaseModel):
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)


class EvaluationQueryOption(BaseModel):
    query_id: str
    media_id: str
    query: str
    duration_sec: float | None = None
    ground_truth: GroundTruthInterval


class EvaluationQueryListResponse(BaseModel):
    manifest_path: str
    total: int
    queries: list[EvaluationQueryOption]


class MomentEvaluationRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    manifest_path: str = Field(
        default="data/manifests/activitynet_dev200_indexed_current.jsonl",
        description="ActivityNet-style manifest containing the selected query and ground truth.",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    window_sec: float = Field(default=10.0, gt=0.0)
    stride_sec: float = Field(default=5.0, gt=0.0)
    profile: str = Field(default="activitynet_visual_only")
    tiou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class MomentEvaluationResult(BaseModel):
    moment: VideoMoment
    tiou: float = Field(ge=0.0, le=1.0)
    hit: bool


class MomentSearchResponse(BaseModel):
    media_id: str
    query: str
    top_k: int
    profile: str
    results: list[VideoMoment]
    total: int


class MomentEvaluationResponse(BaseModel):
    query_id: str
    media_id: str
    query: str
    ground_truth: GroundTruthInterval
    tiou_threshold: float
    hit_rank: int | None = None
    best_tiou: float
    search_response: MomentSearchResponse
    evaluated_results: list[MomentEvaluationResult]


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
