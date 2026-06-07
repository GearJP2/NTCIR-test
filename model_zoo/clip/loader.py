def load_clip():
    import open_clip
    import torch
    from app.core.config import settings
    device = settings.device
    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model,
        pretrained="openai",
        cache_dir=settings.model_cache_dir,
    )
    model = model.to(device).eval()
    return model, preprocess, device
