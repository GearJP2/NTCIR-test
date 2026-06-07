from pathlib import Path

import numpy as np
import structlog

from services.asr.base import ASREngine

logger = structlog.get_logger(__name__)


class Wav2VecEngine(ASREngine):
    """
    HuggingFace wav2vec2 ASR engine — used as a fallback for low-resource
    or domain-specific language variants not well covered by Whisper.
    Note: does not produce word-level timestamps natively; timestamps are
    approximated from CTC alignment.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-large-xlsr-53"):
        self._model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self._model_name,
                return_timestamps="word",
            )
        return self._pipe

    def transcribe(self, audio_path: str | Path, language: str = "th") -> dict:
        pipe = self._load()
        result = pipe(str(audio_path))

        words = []
        for chunk in result.get("chunks", []):
            words.append({
                "word": chunk["text"],
                "start": chunk["timestamp"][0] or 0.0,
                "end": chunk["timestamp"][1] or 0.0,
            })

        logger.debug("asr.wav2vec.done", audio=str(audio_path), words=len(words))
        return {
            "text": result.get("text", "").strip(),
            "language": language,
            "words": words,
        }
