import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad

from app.schemas.media import AudioSegment

_VAD_SAMPLE_RATE = 16000          # webrtcvad requires 16 kHz
_FRAME_DURATION_MS = 30           # 10 | 20 | 30 ms
_AGGRESSIVENESS = 2               # 0–3; higher = more aggressive


def segment_audio(
    audio_path: Path,
    media_id: str,
    min_segment_sec: float = 1.0,
    max_segment_sec: float = 30.0,
) -> list[AudioSegment]:
    """
    VAD-based audio segmentation using webrtcvad.
    Returns a list of AudioSegment objects with trimmed mono 16 kHz wav files.
    """
    import librosa

    audio, sr = librosa.load(str(audio_path), sr=_VAD_SAMPLE_RATE, mono=True)
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

    def _flush(start: float, frames_data: list[bytes]) -> AudioSegment:
        end = start + len(frames_data) * _FRAME_DURATION_MS / 1000
        raw = b"".join(frames_data)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
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

    return segments
