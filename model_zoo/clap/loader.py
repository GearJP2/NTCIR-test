"""CLAP loader via HuggingFace transformers (no laion_clap package required)."""

from __future__ import annotations

CLAP_SR = 48_000


class _HFClapModule:
    """Drop-in replacement for laion_clap.CLAP_Module used by ClapEncoder."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor, ClapModel

        from app.core.config import settings

        self._device = settings.device
        if self._device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"

        cache = settings.model_cache_dir
        model_id = settings.clap_model

        self._processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
        self._model = ClapModel.from_pretrained(model_id, cache_dir=cache)
        self._model = self._model.to(self._device).eval()

    def eval(self) -> _HFClapModule:
        self._model.eval()
        return self

    def get_audio_embedding_from_filelist(self, filepaths, use_tensor: bool = False):
        import numpy as np
        import soundfile as sf
        import torch

        waveforms: list[np.ndarray] = []
        for path in filepaths:
            wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != CLAP_SR:
                new_len = int(len(wav) * CLAP_SR / sr)
                wav = np.interp(
                    np.linspace(0, len(wav) - 1, new_len),
                    np.arange(len(wav)),
                    wav,
                ).astype(np.float32)
            waveforms.append(wav)

        inputs = self._processor(
            audios=[w.tolist() for w in waveforms],
            return_tensors="pt",
            sampling_rate=CLAP_SR,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_audio_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features if use_tensor else features.cpu().numpy()

    def get_text_embedding(self, texts, use_tensor: bool = False):
        import torch

        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features if use_tensor else features.cpu().numpy()


def load_clap() -> _HFClapModule:
    return _HFClapModule().eval()
