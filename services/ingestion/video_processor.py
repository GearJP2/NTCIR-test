import uuid
from pathlib import Path

import av
import numpy as np
from PIL import Image
import soundfile as sf

from app.schemas.media import VideoKeyframe
from configs.model_config import get_ingestion_float

_KEYFRAME_INTERVAL_SEC = 2.0   # extract one keyframe every N seconds


def extract_audio_track(video_path: Path) -> Path:
    """Extract the audio track from a video file to a temp WAV file."""
    out_path = video_path.with_suffix(".audio.wav")
    samples: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        audio_stream = next((stream for stream in container.streams.audio), None)
        if audio_stream is None:
            sf.write(str(out_path), np.array([], dtype=np.float32), 16000)
            return out_path

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                data = resampled.to_ndarray()
                samples.append(data.reshape(-1))
        for resampled in resampler.resample(None):
            data = resampled.to_ndarray()
            samples.append(data.reshape(-1))

    audio = np.concatenate(samples).astype(np.float32) / 32768.0 if samples else np.array([])
    sf.write(str(out_path), audio, 16000)
    return out_path


def extract_keyframes(
    video_path: Path,
    media_id: str,
    interval_sec: float | None = None,
) -> list[VideoKeyframe]:
    """
    Extract one keyframe every `interval_sec` seconds using ffmpeg.
    Returns a list of VideoKeyframe objects pointing to saved JPEG files.
    """
    interval_sec = resolve_keyframe_interval_sec(interval_sec)
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

    out_dir = video_path.parent / "keyframes"
    out_dir.mkdir(exist_ok=True)

    duration = probe_duration_sec(video_path)

    keyframes: list[VideoKeyframe] = []
    timestamps = keyframe_timestamps(duration, interval_sec)
    if not timestamps:
        return keyframes

    next_index = 0
    with av.open(str(video_path)) as container:
        video_stream = next((stream for stream in container.streams.video), None)
        if video_stream is None:
            return keyframes

        for frame in container.decode(video_stream):
            if next_index >= len(timestamps):
                break
            frame_time = float(frame.time or 0.0)
            target_time = timestamps[next_index]
            if frame_time < target_time:
                continue

            frame_id = str(uuid.uuid4())
            out_path = out_dir / f"{frame_id}.jpg"
            image: Image.Image = frame.to_image()
            image.save(out_path, format="JPEG")
            keyframes.append(
                VideoKeyframe(
                    frame_id=frame_id,
                    media_id=media_id,
                    timestamp_sec=target_time,
                    image_path=str(out_path),
                )
            )
            next_index += 1

    return keyframes


def probe_duration_sec(video_path: Path) -> float:
    with av.open(str(video_path)) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)

        video_stream = next((stream for stream in container.streams.video), None)
        if video_stream is not None and video_stream.duration is not None:
            return float(video_stream.duration * video_stream.time_base)

    return 0.0


def keyframe_timestamps(
    duration_sec: float,
    interval_sec: float | None = None,
) -> list[float]:
    """Return keyframe timestamps for baseline visual evidence sampling."""
    interval_sec = resolve_keyframe_interval_sec(interval_sec)
    if duration_sec <= 0:
        return []
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

    timestamps: list[float] = []
    t = 0.0
    while t < duration_sec:
        timestamps.append(round(t, 6))
        t += interval_sec
    return timestamps


def resolve_keyframe_interval_sec(interval_sec: float | None = None) -> float:
    if interval_sec is not None:
        return float(interval_sec)
    return get_ingestion_float("keyframe_interval_sec", _KEYFRAME_INTERVAL_SEC)
