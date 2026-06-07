def load_whisper():
    from app.core.config import settings
    from faster_whisper import WhisperModel
    return WhisperModel(
        settings.whisper_model,
        device=settings.device,
        compute_type="float16" if settings.device == "cuda" else "int8",
        download_root=settings.model_cache_dir,
    )
