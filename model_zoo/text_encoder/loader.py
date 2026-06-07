def load_text_encoder():
    from app.core.config import settings
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.text_encoder_model, cache_folder=settings.model_cache_dir)
