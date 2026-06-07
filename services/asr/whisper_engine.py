from pathlib import Path

import structlog

from services.asr.base import ASREngine

logger = structlog.get_logger(__name__)


class WhisperEngine(ASREngine):
    """
    faster-whisper backed ASR engine with word-level timestamps.
    Model is lazily loaded from the ModelRegistry on first use.
    """

    def transcribe(self, audio_path: str | Path, language: str = "th") -> dict:
        from model_zoo.registry import ModelRegistry
        model = ModelRegistry.get("whisper")

        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
        )

        words = []
        full_text_parts = []
        for seg in segments:
            full_text_parts.append(seg.text)
            if seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})

        logger.debug(
            "asr.whisper.done",
            audio=str(audio_path),
            detected_lang=info.language,
            words=len(words),
        )
        return {
            "text": " ".join(full_text_parts).strip(),
            "language": info.language,
            "words": words,
        }
