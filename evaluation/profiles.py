from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from services.retrieval.moments import SourceType

SearchScope = Literal["single_video"]

REQUIRED_SOURCE_TYPES: tuple[SourceType, ...] = ("visual", "asr", "audio", "summary")
DEFAULT_PROFILE_PATH = Path("configs/evaluation_profiles.yaml")


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    search_scope: SearchScope
    top_k: int
    tiou_threshold: float | None
    modality_weights: dict[SourceType, float]


def load_evaluation_profiles(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, EvaluationProfile]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{path} must define a non-empty 'profiles' mapping")

    return {
        str(name): _parse_profile(str(name), data)
        for name, data in profiles.items()
    }


def get_evaluation_profile(
    name: str,
    path: Path = DEFAULT_PROFILE_PATH,
) -> EvaluationProfile:
    profiles = load_evaluation_profiles(path)
    try:
        return profiles[name]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown Evaluation Profile '{name}'. Available: {available}") from exc


def _parse_profile(name: str, data: object) -> EvaluationProfile:
    if not isinstance(data, dict):
        raise ValueError(f"Profile '{name}' must be a mapping")

    search_scope = str(data.get("search_scope", ""))
    if search_scope != "single_video":
        raise ValueError(f"Profile '{name}' has unsupported search_scope={search_scope!r}")

    top_k = int(data.get("top_k", 0))
    if top_k <= 0:
        raise ValueError(f"Profile '{name}' must set top_k > 0")

    tiou = data.get("tiou_threshold")
    tiou_threshold = float(tiou) if tiou is not None else None
    if tiou_threshold is not None and not 0.0 <= tiou_threshold <= 1.0:
        raise ValueError(f"Profile '{name}' tiou_threshold must be between 0 and 1")

    weights = _parse_weights(name, data.get("modality_weights"))

    return EvaluationProfile(
        name=name,
        search_scope="single_video",
        top_k=top_k,
        tiou_threshold=tiou_threshold,
        modality_weights=weights,
    )


def _parse_weights(name: str, data: object) -> dict[SourceType, float]:
    if not isinstance(data, dict):
        raise ValueError(f"Profile '{name}' must define modality_weights")

    missing = [source_type for source_type in REQUIRED_SOURCE_TYPES if source_type not in data]
    if missing:
        raise ValueError(f"Profile '{name}' missing modality weights: {', '.join(missing)}")

    weights: dict[SourceType, float] = {}
    for source_type in REQUIRED_SOURCE_TYPES:
        value = float(data[source_type])
        if value < 0.0:
            raise ValueError(f"Profile '{name}' has negative weight for {source_type}")
        weights[source_type] = value

    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"Profile '{name}' modality weights must sum to a positive value")

    return weights
