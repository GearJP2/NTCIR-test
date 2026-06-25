from types import SimpleNamespace

from av.error import InvalidDataError

from scripts import sample_castle_frames
from scripts.sample_castle_frames import _decode_remote_frame, sample_timestamps


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


def test_decode_remote_frame_reopens_after_transient_decoder_failure(
    monkeypatch,
):
    expected_frame = SimpleNamespace(time=10.0)
    attempts = iter(
        [
            InvalidDataError(1, "transient"),
            [expected_frame],
        ]
    )

    class FakeContainer:
        streams = [SimpleNamespace(type="video", time_base=1.0)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def seek(self, *_args, **_kwargs):
            return None

        def decode(self, _stream):
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            return iter(result)

    monkeypatch.setattr(
        sample_castle_frames.av,
        "open",
        lambda *_args, **_kwargs: FakeContainer(),
    )

    assert _decode_remote_frame("remote.mp4", 10.0, max_attempts=2) is expected_frame
