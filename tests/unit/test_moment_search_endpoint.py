import pytest
from fastapi import HTTPException

from app.api.v1.endpoints.search import search_moments
from app.schemas.search import MomentSearchRequest


@pytest.mark.asyncio
async def test_search_moments_endpoint_returns_empty_baseline():
    request = MomentSearchRequest(
        media_id="v_123",
        query="woman doing sit ups",
        top_k=10,
    )

    response = await search_moments(request)

    assert response.media_id == "v_123"
    assert response.query == "woman doing sit ups"
    assert response.top_k == 10
    assert response.profile == "activitynet_visual_heavy"
    assert response.results == []
    assert response.total == 0


@pytest.mark.asyncio
async def test_search_moments_endpoint_rejects_unknown_profile():
    request = MomentSearchRequest(
        media_id="v_123",
        query="woman doing sit ups",
        profile="missing",
    )

    with pytest.raises(HTTPException) as exc_info:
        await search_moments(request)

    assert exc_info.value.status_code == 422
    assert "Unknown Evaluation Profile" in exc_info.value.detail
