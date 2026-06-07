from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    """Abstract base for all embedding encoders."""

    @abstractmethod
    def encode(self, input_: str) -> np.ndarray:
        """Encode a single input to a float32 vector."""
        ...

    def encode_batch(self, inputs: list[str]) -> list[np.ndarray]:
        """Default: loop over encode(). Override for batched GPU inference."""
        return [self.encode(x) for x in inputs]
