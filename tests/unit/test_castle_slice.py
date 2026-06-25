import pytest

from services.dataset.castle_slice import analyze_transcript, select_recording_paths


def test_analyze_transcript_detects_invalid_intervals_and_order():
    quality = analyze_transcript(
        {
            "chunks": [
                {"timestamp": [0, 2], "text": "hello"},
                {"timestamp": [4, 3], "text": ""},
                {"timestamp": [1, 5], "text": "later in file"},
            ]
        }
    )

    assert quality.chunk_count == 3
    assert quality.valid_chunk_count == 2
    assert quality.empty_chunk_count == 1
    assert quality.reversed_interval_count == 1
    assert quality.non_monotonic_count == 1


def test_analyze_transcript_requires_chunks_list():
    with pytest.raises(ValueError, match="chunks list"):
        analyze_transcript({"text": "missing chunks"})


def test_select_recording_paths_pairs_sources_and_ignores_checksums():
    selected = select_recording_paths(
        [
            "main/day1/Allie/video/08.mp4",
            "main/day1/Allie/video/08.mp4.sha256",
            "main/day1/Allie/transcript/08.json",
            "main/day1/Allie/metadata/08.ACCL.csv",
            "main/day2/Allie/video/08.mp4",
        ],
        day="day1",
        participant_id="Allie",
    )

    assert selected == {
        "08": {
            "video": "main/day1/Allie/video/08.mp4",
            "transcript": "main/day1/Allie/transcript/08.json",
            "metadata_prefix": "08",
        }
    }
