from scripts.sample_castle_frames import sample_timestamps


def test_sample_timestamps_stops_before_duration():
    assert sample_timestamps(3_600_039, 600) == [
        0,
        600,
        1200,
        1800,
        2400,
        3000,
        3600,
    ]


def test_sample_timestamps_supports_bounded_interval():
    assert sample_timestamps(
        3_600_000,
        5,
        start_sec=60,
        end_sec=75,
    ) == [60, 65, 70]
