"""
ModelRegistry — lazy singleton cache for all heavy ML models.

Usage:
    from model_zoo.registry import ModelRegistry
    model = ModelRegistry.get("whisper")
"""

import threading
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_registry: dict[str, Any] = {}
_lock = threading.Lock()

_LOADERS: dict[str, str] = {
    "whisper":       "model_zoo.whisper.loader:load_whisper",
    "clap":          "model_zoo.clap.loader:load_clap",
    "clip":          "model_zoo.clip.loader:load_clip",
    "text_encoder":  "model_zoo.text_encoder.loader:load_text_encoder",
    "reranker":      "model_zoo.reranker.loader:load_reranker",
}


class ModelRegistry:
    @classmethod
    def get(cls, name: str) -> Any:
        if name not in _registry:
            with _lock:
                if name not in _registry:  # double-checked locking
                    _registry[name] = cls._load(name)
        return _registry[name]

    @staticmethod
    def _load(name: str) -> Any:
        if name not in _LOADERS:
            raise KeyError(f"Unknown model key: '{name}'. Available: {list(_LOADERS)}")
        module_path, func_name = _LOADERS[name].rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        loader_fn = getattr(module, func_name)
        logger.info("model_zoo.loading", model=name)
        model = loader_fn()
        logger.info("model_zoo.loaded", model=name)
        return model

    @classmethod
    def preload_all(cls) -> None:
        """Call at worker startup to warm all models before serving requests."""
        for name in _LOADERS:
            cls.get(name)
