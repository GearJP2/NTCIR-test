import pytest

from configs.model_config import get_ingestion_float, load_model_config


def test_load_model_config_reads_default_yaml():
    config = load_model_config()

    assert config["ingestion"]["keyframe_interval_sec"] == 2.0


def test_get_ingestion_float_reads_value_from_yaml(tmp_path):
    path = tmp_path / "model_config.yaml"
    path.write_text(
        """
ingestion:
  keyframe_interval_sec: 1.5
""",
        encoding="utf-8",
    )

    assert get_ingestion_float("keyframe_interval_sec", 2.0, path=path) == 1.5


def test_get_ingestion_float_uses_default_for_missing_key(tmp_path):
    path = tmp_path / "model_config.yaml"
    path.write_text("ingestion: {}\n", encoding="utf-8")

    assert get_ingestion_float("missing", 3.0, path=path) == 3.0


def test_get_ingestion_float_rejects_non_numeric_value(tmp_path):
    path = tmp_path / "model_config.yaml"
    path.write_text(
        """
ingestion:
  keyframe_interval_sec: nope
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a number"):
        get_ingestion_float("keyframe_interval_sec", 2.0, path=path)
