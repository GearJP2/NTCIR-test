from abc import ABC, abstractmethod
from pathlib import Path


class ASREngine(ABC):
    """Abstract base for all ASR backends."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path, language: str = "th") -> dict:
        """
        Transcribe audio to text with word-level timestamps.

        Returns:
            {
                "text": str,
                "language": str,
                "words": [{"word": str, "start": float, "end": float}]
            }
        """
        ...
