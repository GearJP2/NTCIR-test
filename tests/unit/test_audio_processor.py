import numpy as np
import soundfile as sf

from services.ingestion.audio_processor import load_audio_mono


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
