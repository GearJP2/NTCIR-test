"""
Search endpoints

POST /api/v1/search/episodic  — Primary: text-to-audio semantic search over
                                `csat_episodic_memory` with optional WorldMM
                                LLM reasoning.

POST /api/v1/search/          — Legacy: multimodal ANN search across
                                audio_segments / visual_keyframes / text_transcripts.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from app.schemas.search import (
    EpisodicSearchRequest,
    EpisodicSearchResponse,
    SearchRequest,
    SearchResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Primary endpoint: Episodic memory search ──────────────────────────────────

@router.post(
    "/episodic",
    response_model=EpisodicSearchResponse,
    summary="Semantic search over episodic audio memory",
    description=(
        "Encodes the natural-language query into the same vector space used "
        "at ingest time (CLAP or Wav2Vec2), searches the `csat_episodic_memory` "
        "Milvus collection for the Top-K nearest audio chunks, then optionally "
        "runs a WorldMM-style LLM agent to reason over the retrieved context and "
        "pinpoint the exact interaction event."
    ),
)
async def search_episodic(request: EpisodicSearchRequest) -> EpisodicSearchResponse:
    """
    **WorldMM-inspired episodic memory retrieval pipeline:**

    1. Text query → CLAP / sentence-transformer vector  
    2. ANN search in `csat_episodic_memory` (cosine similarity, HNSW)  
    3. Presigned MinIO URLs hydrated on each hit  
    4. (optional) WorldMM prompt → LLM → structured `LLMReasoning` response
    """
    from services.query_service import QueryService

    try:
        service = QueryService(embedder_name=request.embedder)
        response = await service.run(
            query=request.query,
            top_k=request.top_k,
            media_id_filter=request.media_id_filter,
            score_threshold=request.score_threshold,
            use_llm=request.use_llm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("search.episodic.runtime_error", query=request.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("search.episodic.unhandled", query=request.query)
        raise HTTPException(status_code=500, detail="Search failed unexpectedly.") from exc

    logger.info(
        "search.episodic.done",
        query=request.query[:80],
        hits=response.total_hits,
        use_llm=request.use_llm,
        embedder=request.embedder,
        top_k=request.top_k,
    )
    return response


# ── Legacy endpoint: Multimodal ANN search ────────────────────────────────────

@router.post(
    "/",
    response_model=SearchResponse,
    summary="Legacy multimodal ANN search",
    description=(
        "Searches across audio_segments, visual_keyframes, and text_transcripts "
        "collections using RRF fusion and cross-encoder reranking. "
        "Prefer `/search/episodic` for CSAT episodic memory retrieval."
    ),
)
async def search_multimodal(request: SearchRequest) -> SearchResponse:
    from services.retrieval.searcher import multimodal_search

    hits = await multimodal_search(
        text_query=request.text_query,
        audio_url=request.audio_url,
        top_k=request.top_k,
        modalities=request.modalities,
    )
    logger.info("search.multimodal.done", query=request.text_query, hits=len(hits))
    return SearchResponse(results=hits, total=len(hits))
