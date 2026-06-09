from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_CONFIG_PATH = Path(__file__).with_name("model_config.yaml")


@lru_cache(maxsize=1)
def load_model_config(path: Path = DEFAULT_MODEL_CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def get_ingestion_float(
    key: str,
    default: float,
    path: Path = DEFAULT_MODEL_CONFIG_PATH,
) -> float:
    config = load_model_config(path)
    ingestion = config.get("ingestion", {})
    if not isinstance(ingestion, dict):
        raise ValueError(f"{path} ingestion section must be a mapping")

    value = ingestion.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} ingestion.{key} must be a number") from exc
