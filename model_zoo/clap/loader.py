def load_clap():
    import laion_clap
    from app.core.config import settings
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt()          # downloads to HF cache / model_cache_dir
    model.eval()
    return model
