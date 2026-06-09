from __future__ import annotations

import structlog

from app.schemas.search import MomentSearchRequest, MomentSearchResponse
from evaluation.profiles import get_evaluation_profile

logger = structlog.get_logger(__name__)


class MomentSearchService:
    """
    Canonical long-video search entry point.

    This service owns the benchmark-facing contract:
    Semantic Query + selected media ID -> ranked Video Moments. The current
    implementation is an empty baseline until fixed-window indexing and
    per-modality late fusion are wired in.
    """

    async def run(self, request: MomentSearchRequest) -> MomentSearchResponse:
        profile = get_evaluation_profile(request.profile)
        logger.info(
            "moment_search.empty_baseline",
            media_id=request.media_id,
            query=request.query[:80],
            top_k=request.top_k,
            profile=profile.name,
        )
        return MomentSearchResponse(
            media_id=request.media_id,
            query=request.query,
            top_k=request.top_k,
            profile=profile.name,
            results=[],
            total=0,
        )
