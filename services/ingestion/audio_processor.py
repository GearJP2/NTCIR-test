import uuid
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad
from scipy import signal

from app.schemas.media import AudioSegment

_VAD_SAMPLE_RATE = 16000          # webrtcvad requires 16 kHz
_FRAME_DURATION_MS = 30           # 10 | 20 | 30 ms
_AGGRESSIVENESS = 2               # 0–3; higher = more aggressive


def segment_audio(
    audio_path: Path,
    media_id: str,
    min_segment_sec: float = 1.0,
    max_segment_sec: float = 30.0,
    fallback_to_full_track: bool = True,
) -> list[AudioSegment]:
    """
    VAD-based audio segmentation using webrtcvad.
    Returns a list of AudioSegment objects with trimmed mono 16 kHz wav files.
    """
    audio = load_audio_mono(audio_path, _VAD_SAMPLE_RATE)
    pcm = (audio * 32768).astype(np.int16).tobytes()

    vad = webrtcvad.Vad(_AGGRESSIVENESS)
    frame_len = int(_VAD_SAMPLE_RATE * _FRAME_DURATION_MS / 1000)
    frame_bytes = frame_len * 2  # 16-bit = 2 bytes/sample

    frames = [pcm[i: i + frame_bytes] for i in range(0, len(pcm) - frame_bytes, frame_bytes)]
    is_speech = [vad.is_speech(f, _VAD_SAMPLE_RATE) for f in frames if len(f) == frame_bytes]

    # Merge contiguous speech frames into segments
    segments: list[AudioSegment] = []
    in_segment = False
    seg_start = 0.0
    seg_frames: list[bytes] = []

    out_dir = audio_path.parent / "segments"
    out_dir.mkdir(exist_ok=True)

    def _write_segment(start: float, samples: np.ndarray) -> AudioSegment:
        end = start + len(samples) / _VAD_SAMPLE_RATE
        seg_id = str(uuid.uuid4())
        out_path = out_dir / f"{seg_id}.wav"
        sf.write(str(out_path), samples, _VAD_SAMPLE_RATE)
        return AudioSegment(
            segment_id=seg_id,
            media_id=media_id,
            start_sec=start,
            end_sec=end,
            audio_path=str(out_path),
        )

    def _flush(start: float, frames_data: list[bytes]) -> AudioSegment:
        raw = b"".join(frames_data)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
        return _write_segment(start, samples)

    for idx, speech in enumerate(is_speech):
        t = idx * _FRAME_DURATION_MS / 1000
        if speech:
            if not in_segment:
                in_segment = True
                seg_start = t
                seg_frames = []
            seg_frames.append(frames[idx])

            # Force-flush at max_segment_sec
            if (len(seg_frames) * _FRAME_DURATION_MS / 1000) >= max_segment_sec:
                segments.append(_flush(seg_start, seg_frames))
                seg_start = t + _FRAME_DURATION_MS / 1000
                seg_frames = []
        else:
            if in_segment:
                duration = len(seg_frames) * _FRAME_DURATION_MS / 1000
                if duration >= min_segment_sec:
                    segments.append(_flush(seg_start, seg_frames))
                in_segment = False
                seg_frames = []

    if in_segment and seg_frames:
        segments.append(_flush(seg_start, seg_frames))

    if not segments and fallback_to_full_track and audio.size:
        segments.extend(_fallback_segments(audio, max_segment_sec, _write_segment))

    return segments


def _fallback_segments(
    audio: np.ndarray,
    max_segment_sec: float,
    write_segment,
) -> list[AudioSegment]:
    """Create fixed windows when VAD finds no speech, preserving non-speech audio evidence."""
    segment_samples = max(1, int(max_segment_sec * _VAD_SAMPLE_RATE))
    segments: list[AudioSegment] = []
    for start_sample in range(0, len(audio), segment_samples):
        samples = audio[start_sample: start_sample + segment_samples]
        if not samples.size:
            continue
        start_sec = start_sample / _VAD_SAMPLE_RATE
        segments.append(write_segment(start_sec, samples))
    return segments


def load_audio_mono(audio_path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        divisor = gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // divisor, sr // divisor)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
