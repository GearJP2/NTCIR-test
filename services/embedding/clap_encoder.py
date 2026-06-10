from pathlib import Path

import numpy as np
import structlog

from services.embedding.base import Encoder

logger = structlog.get_logger(__name__)

_AUDIO_DIM = 512


class ClapEncoder(Encoder):
    """
    LAION-CLAP encoder for audio and text.
    Produces 512-dim L2-normalised embeddings.
    Audio and text share the same embedding space — enabling cross-modal search.
    """

    def encode(self, input_: str) -> np.ndarray:
        """Encode a single audio file path."""
        return self.encode_batch([input_])[0]

    def encode_batch(self, inputs: list[str]) -> list[np.ndarray]:
        """Encode a batch of audio file paths."""
        if not inputs:
            return []

        from model_zoo.registry import ModelRegistry
        model = ModelRegistry.get("clap")

        embeddings = model.get_audio_embedding_from_filelist(inputs, use_tensor=False)
        # Normalise to unit sphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        return list(embeddings.astype(np.float32))

    def encode_text(self, texts: list[str]) -> list[np.ndarray]:
        """Encode text queries into the shared CLAP embedding space."""
        from model_zoo.registry import ModelRegistry
        model = ModelRegistry.get("clap")
        embeddings = model.get_text_embedding(texts, use_tensor=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        return list(embeddings.astype(np.float32))
