import numpy as np
import structlog

from services.embedding.base import Encoder

logger = structlog.get_logger(__name__)

_TEXT_DIM = 768


class TextEncoder(Encoder):
    """
    sentence-transformers dense text encoder for transcript chunks.
    Used for the text Milvus collection and for semantic BM25 hybrid search.
    """

    def encode(self, input_: str) -> np.ndarray:
        return self.encode_batch([input_])[0]

    def encode_batch(self, inputs: list[str]) -> list[np.ndarray]:
        from model_zoo.registry import ModelRegistry
        model = ModelRegistry.get("text_encoder")
        embeddings = model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
        return list(embeddings.astype(np.float32))
