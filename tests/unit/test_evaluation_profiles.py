import pytest

from evaluation.profiles import get_evaluation_profile, load_evaluation_profiles


def test_load_default_evaluation_profiles():
    profiles = load_evaluation_profiles()

    activitynet = profiles["activitynet_visual_heavy"]
    assert activitynet.search_scope == "single_video"
    assert activitynet.top_k == 10
    assert activitynet.tiou_threshold == 0.3
    assert activitynet.modality_weights["visual"] == 0.60
    assert activitynet.modality_weights["asr"] == 0.25

    castle = profiles["castle_lifelog_balanced"]
    assert castle.tiou_threshold is None
    assert castle.modality_weights["visual"] == 0.35


def test_get_evaluation_profile_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown Evaluation Profile"):
        get_evaluation_profile("missing")


def test_load_evaluation_profiles_rejects_missing_weight(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        """
profiles:
  bad:
    search_scope: single_video
    top_k: 10
    tiou_threshold: 0.3
    modality_weights:
      visual: 1.0
      asr: 0.0
      audio: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing modality weights: summary"):
        load_evaluation_profiles(path)


def test_load_evaluation_profiles_rejects_invalid_tiou(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text(
        """
profiles:
  bad:
    search_scope: single_video
    top_k: 10
    tiou_threshold: 1.5
    modality_weights:
      visual: 1.0
      asr: 0.0
      audio: 0.0
      summary: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tiou_threshold"):
        load_evaluation_profiles(path)
