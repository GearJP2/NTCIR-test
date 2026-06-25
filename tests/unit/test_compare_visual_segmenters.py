from scripts.compare_visual_segmenters import _median_step


def test_median_step_uses_sample_timestamps():
    assert _median_step([0, 5_000, 10_000, 20_000]) == 5_000
