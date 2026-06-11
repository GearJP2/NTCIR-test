import json
from pathlib import Path

from scripts.compare_moment_results import compare_results
from scripts.summarize_evaluation_results import summarize_results


def test_summarize_results_writes_csv_and_markdown(tmp_path: Path):
    summary_a = tmp_path / "a_summary.json"
    summary_b = tmp_path / "sweep_summary.json"
    summary_a.write_text(
        json.dumps(
            {
                "profile": "activitynet_visual_only",
                "top_k": 10,
                "tiou_threshold": 0.3,
                "num_videos": 2,
                "num_queries": 3,
                "Recall@10": 0.5,
                "mAP@10": 0.25,
            }
        ),
        encoding="utf-8",
    )
    summary_b.write_text(
        json.dumps(
            [
                {
                    "profile": "activitynet_visual_heavy",
                    "top_k": 10,
                    "tiou_threshold": 0.3,
                    "num_videos": 2,
                    "num_queries": 3,
                    "Recall@10": 0.4,
                    "mAP@10": 0.2,
                },
                {
                    "profile": "activitynet_visual_only",
                    "top_k": 10,
                    "tiou_threshold": 0.3,
                    "num_videos": 2,
                    "num_queries": 3,
                    "Recall@10": 0.5,
                    "mAP@10": 0.25,
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = summarize_results(
        [summary_a, summary_b],
        csv_path=tmp_path / "table.csv",
        markdown_path=tmp_path / "table.md",
    )

    assert [row["profile"] for row in rows] == [
        "activitynet_visual_heavy",
        "activitynet_visual_only",
    ]
    assert "activitynet_visual_only" in (tmp_path / "table.csv").read_text(encoding="utf-8")
    assert "| activitynet_visual_heavy |" in (tmp_path / "table.md").read_text(encoding="utf-8")


def test_compare_results_classifies_regressions_and_improvements(tmp_path: Path):
    baseline = tmp_path / "baseline_results.jsonl"
    candidate = tmp_path / "candidate_results.jsonl"
    baseline_rows = [
        _query_row("q1", hit=True, hit_rank=1, best_tiou=0.7),
        _query_row("q2", hit=False, hit_rank=None, best_tiou=0.1),
        _query_row("q3", hit=True, hit_rank=2, best_tiou=0.5),
    ]
    candidate_rows = [
        _query_row("q1", hit=False, hit_rank=None, best_tiou=0.2),
        _query_row("q2", hit=True, hit_rank=3, best_tiou=0.4),
        _query_row("q3", hit=True, hit_rank=1, best_tiou=0.6),
    ]
    baseline.write_text("\n".join(json.dumps(row) for row in baseline_rows) + "\n", encoding="utf-8")
    candidate.write_text("\n".join(json.dumps(row) for row in candidate_rows) + "\n", encoding="utf-8")

    result = compare_results(
        baseline_results_path=baseline,
        candidate_results_path=candidate,
        csv_path=tmp_path / "regressions.csv",
        markdown_path=tmp_path / "regressions.md",
        json_path=tmp_path / "regressions.json",
    )

    summary = result["summary"]
    assert summary["regressions"] == 1
    assert summary["improvements"] == 1
    assert summary["rank_improvements"] == 1
    assert "regression" in (tmp_path / "regressions.csv").read_text(encoding="utf-8")
    assert "Lost Hits" in (tmp_path / "regressions.md").read_text(encoding="utf-8")


def _query_row(query_id: str, hit: bool, hit_rank: int | None, best_tiou: float) -> dict:
    result_hit = hit_rank == 1
    return {
        "query_id": query_id,
        "media_id": "v_1",
        "query": f"query {query_id}",
        "ground_truth": {"start_sec": 1.0, "end_sec": 5.0},
        "top_k": 10,
        "tiou_threshold": 0.3,
        "hit": hit,
        "hit_rank": hit_rank,
        "best_tiou": best_tiou,
        "results": [
            {
                "rank": 1,
                "moment_id": "v_1:0.000-10.000",
                "media_id": "v_1",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "score": 0.9,
                "tiou": best_tiou,
                "hit": result_hit,
            }
        ],
    }
