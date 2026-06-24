import json
from pathlib import Path

from scripts.estimate_activitynet_ablation_cost import estimate_ablation_costs


def test_estimate_ablation_costs_writes_tables(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "media_id": "v_1",
                "video_path": "videos/v_1.mp4",
                "duration_sec": 20.0,
                "queries": [
                    {
                        "query_id": "v_1:0",
                        "query": "query one",
                        "ground_truth": {"start_sec": 0.0, "end_sec": 5.0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = estimate_ablation_costs(
        manifest_path=manifest,
        output_csv=tmp_path / "costs.csv",
        output_markdown=tmp_path / "costs.md",
        output_json=tmp_path / "costs.json",
        window_stride=["10:5", "20:10"],
        keyframe_interval=[2.0, 10.0],
    )

    by_setting = {(row["ablation_type"], row["setting"]): row for row in rows}
    assert by_setting[("moment_windows", "10s/5s")]["total_units"] == 3
    assert by_setting[("moment_windows", "20s/10s")]["total_units"] == 1
    assert by_setting[("visual_keyframes", "2s")]["total_units"] == 10
    assert by_setting[("visual_keyframes", "10s")]["relative_to_default"] == 0.2
    assert "ActivityNet Ablation Cost Estimates" in (tmp_path / "costs.md").read_text(
        encoding="utf-8"
    )


def test_estimate_ablation_costs_uses_first_custom_setting_as_relative_baseline(
    tmp_path: Path,
):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "media_id": "v_1",
                "video_path": "videos/v_1.mp4",
                "duration_sec": 20.0,
                "queries": [
                    {
                        "query_id": "v_1:0",
                        "query": "query one",
                        "ground_truth": {"start_sec": 0.0, "end_sec": 5.0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = estimate_ablation_costs(
        manifest_path=manifest,
        output_csv=tmp_path / "costs.csv",
        output_markdown=tmp_path / "costs.md",
        window_stride=["20:10"],
        keyframe_interval=[10.0],
    )

    assert [row["relative_to_default"] for row in rows] == [1.0, 1.0]
