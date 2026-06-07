import numpy as np

# Fusion weights — tune these on your dev qrels
_WEIGHT_AUDIO = 0.4
_WEIGHT_VISUAL = 0.3
_WEIGHT_TEXT = 0.3


def fuse_embeddings(
    audio_vec: np.ndarray | None,
    visual_vec: np.ndarray | None,
    text_vec: np.ndarray | None,
) -> np.ndarray:
    """
    Late fusion via weighted concatenation of available modality vectors.
    Absent modalities are skipped and weights are renormalised automatically.
    The resulting vector is L2-normalised.
    """
    parts: list[np.ndarray] = []
    total_weight = 0.0

    if audio_vec is not None:
        parts.append(audio_vec * _WEIGHT_AUDIO)
        total_weight += _WEIGHT_AUDIO
    if visual_vec is not None:
        parts.append(visual_vec * _WEIGHT_VISUAL)
        total_weight += _WEIGHT_VISUAL
    if text_vec is not None:
        parts.append(text_vec * _WEIGHT_TEXT)
        total_weight += _WEIGHT_TEXT

    if not parts:
        raise ValueError("At least one modality vector must be provided for fusion.")

    # Renormalise weights so they sum to 1
    fused = np.concatenate([p / total_weight for p in parts])
    norm = np.linalg.norm(fused)
    return (fused / max(norm, 1e-8)).astype(np.float32)
