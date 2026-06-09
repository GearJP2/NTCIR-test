import pytest

from configs.model_config import load_model_config
from services.ingestion.video_processor import keyframe_timestamps, resolve_keyframe_interval_sec


def test_keyframe_timestamps_defaults_to_two_second_interval():
    load_model_config.cache_clear()

    assert keyframe_timestamps(10.0) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_keyframe_timestamps_supports_custom_interval():
    assert keyframe_timestamps(5.0, interval_sec=1.5) == [0.0, 1.5, 3.0, 4.5]


def test_keyframe_timestamps_returns_empty_for_non_positive_duration():
    assert keyframe_timestamps(0.0) == []
    assert keyframe_timestamps(-1.0) == []


def test_keyframe_timestamps_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="interval_sec must be positive"):
        keyframe_timestamps(10.0, interval_sec=0.0)


def test_resolve_keyframe_interval_accepts_explicit_override():
    assert resolve_keyframe_interval_sec(1.25) == 1.25
