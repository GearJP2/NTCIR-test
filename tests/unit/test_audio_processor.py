import numpy as np
import soundfile as sf

from services.ingestion.audio_processor import load_audio_mono, segment_audio


def test_load_audio_mono_converts_stereo_and_resamples(tmp_path):
    sr = 8000
    left = np.ones(sr, dtype=np.float32) * 0.25
    right = np.ones(sr, dtype=np.float32) * -0.25
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "audio.wav"
    sf.write(path, stereo, sr)

    audio = load_audio_mono(path, target_sr=16000)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 16000
    assert np.max(np.abs(audio)) < 1e-4


def test_segment_audio_falls_back_to_fixed_windows_when_vad_finds_no_speech(tmp_path):
    sr = 16000
    silence = np.zeros(int(sr * 2.5), dtype=np.float32)
    path = tmp_path / "silent.wav"
    sf.write(path, silence, sr)

    segments = segment_audio(path, "media-1", max_segment_sec=1.0)

    assert len(segments) == 3
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 1.0
    assert segments[1].start_sec == 1.0
    assert segments[1].end_sec == 2.0
    assert segments[2].start_sec == 2.0
    assert segments[2].end_sec == 2.5
    assert all(segment.media_id == "media-1" for segment in segments)


def test_segment_audio_can_disable_no_speech_fallback(tmp_path):
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    path = tmp_path / "silent.wav"
    sf.write(path, silence, sr)

    segments = segment_audio(path, "media-1", fallback_to_full_track=False)

    assert segments == []
