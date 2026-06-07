import uuid
from pathlib import Path

import ffmpeg
from PIL import Image

from app.schemas.media import VideoKeyframe

_KEYFRAME_INTERVAL_SEC = 5.0   # extract one keyframe every N seconds


def extract_audio_track(video_path: Path) -> Path:
    """Extract the audio track from a video file to a temp WAV file."""
    out_path = video_path.with_suffix(".audio.wav")
    (
        ffmpeg
        .input(str(video_path))
        .output(str(out_path), ac=1, ar=16000, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path


def extract_keyframes(video_path: Path, media_id: str) -> list[VideoKeyframe]:
    """
    Extract one keyframe every `_KEYFRAME_INTERVAL_SEC` seconds using ffmpeg.
    Returns a list of VideoKeyframe objects pointing to saved JPEG files.
    """
    out_dir = video_path.parent / "keyframes"
    out_dir.mkdir(exist_ok=True)

    # Probe duration
    probe = ffmpeg.probe(str(video_path))
    duration = float(probe["format"]["duration"])

    keyframes: list[VideoKeyframe] = []
    t = 0.0
    while t < duration:
        frame_id = str(uuid.uuid4())
        out_path = out_dir / f"{frame_id}.jpg"
        (
            ffmpeg
            .input(str(video_path), ss=t)
            .output(str(out_path), vframes=1, format="image2", vcodec="mjpeg")
            .overwrite_output()
            .run(quiet=True)
        )
        if out_path.exists():
            keyframes.append(
                VideoKeyframe(
                    frame_id=frame_id,
                    media_id=media_id,
                    timestamp_sec=t,
                    image_path=str(out_path),
                )
            )
        t += _KEYFRAME_INTERVAL_SEC

    return keyframes
