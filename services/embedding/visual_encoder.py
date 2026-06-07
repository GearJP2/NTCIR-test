import numpy as np
import structlog
from PIL import Image

from services.embedding.base import Encoder

logger = structlog.get_logger(__name__)

_VISUAL_DIM = 768


class VisualEncoder(Encoder):
    """
    CLIP ViT-L/14 keyframe encoder.
    Produces 768-dim L2-normalised image embeddings.
    """

    def encode(self, input_: str) -> np.ndarray:
        return self.encode_batch([input_])[0]

    def encode_batch(self, inputs: list[str]) -> list[np.ndarray]:
        import torch
        from model_zoo.registry import ModelRegistry

        model, preprocess, device = ModelRegistry.get("clip")

        images = [preprocess(Image.open(p).convert("RGB")) for p in inputs]
        batch = torch.stack(images).to(device)

        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)

        return list(features.cpu().numpy().astype(np.float32))

    def encode_text(self, texts: list[str]) -> list[np.ndarray]:
        """Encode text queries into CLIP embedding space for visual search."""
        import open_clip
        import torch
        from model_zoo.registry import ModelRegistry

        model, _, device = ModelRegistry.get("clip")
        tokens = open_clip.tokenize(texts).to(device)

        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return list(features.cpu().numpy().astype(np.float32))
