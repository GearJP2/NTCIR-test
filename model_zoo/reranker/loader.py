def load_reranker():
    from app.core.config import settings
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model, max_length=512)
