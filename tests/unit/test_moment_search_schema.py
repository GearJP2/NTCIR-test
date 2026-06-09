import pytest
from pydantic import ValidationError

from app.schemas.search import Evidence, MomentSearchRequest, VideoMoment


def test_moment_search_request_defaults_to_activitynet_profile():
    request = MomentSearchRequest(media_id="v_123", query="woman doing sit ups")

    assert request.top_k == 10
    assert request.profile == "activitynet_visual_heavy"


def test_video_moment_carries_source_specific_evidence():
    moment = VideoMoment(
        rank=1,
        moment_id="v_123:40.000-50.000",
        media_id="v_123",
        start_sec=40.0,
        end_sec=50.0,
        score=0.82,
        thumbnail_sec=44.0,
        evidence=[
            Evidence(
                source_type="visual",
                score=0.78,
                source_id="frame-abc",
                timestamp_sec=44.0,
            )
        ],
    )

    assert moment.evidence[0].source_type == "visual"
    assert moment.evidence[0].timestamp_sec == 44.0


def test_evidence_rejects_unknown_source_type():
    with pytest.raises(ValidationError):
        Evidence(source_type="caption", score=1.0)
