import json
from pathlib import Path

from scripts.compare_moment_results import compare_results
from scripts.check_activitynet_paper_artifacts import check_activitynet_paper_artifacts
from scripts.summarize_activitynet_temporal_tradeoff import summarize_temporal_tradeoff
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
        latex_path=tmp_path / "table.tex",
        findings_path=tmp_path / "findings.md",
    )

    assert [row["profile"] for row in rows] == [
        "activitynet_visual_heavy",
        "activitynet_visual_only",
    ]
    assert "activitynet_visual_only" in (tmp_path / "table.csv").read_text(encoding="utf-8")
    assert "| activitynet_visual_heavy |" in (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "\\begin{table}" in (tmp_path / "table.tex").read_text(encoding="utf-8")
    assert "activitynet\\_visual\\_only" in (tmp_path / "table.tex").read_text(encoding="utf-8")
    assert "Paper-Ready Sentence" in (tmp_path / "findings.md").read_text(encoding="utf-8")


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


def test_summarize_temporal_tradeoff_writes_quality_cost_table(tmp_path: Path):
    summary_10_5 = tmp_path / "summary_10_5.json"
    summary_20_10 = tmp_path / "summary_20_10.json"
    costs = tmp_path / "costs.json"
    summary_10_5.write_text(
        json.dumps(
            {
                "profile": "activitynet_visual_only",
                "top_k": 10,
                "window_sec": 10.0,
                "stride_sec": 5.0,
                "tiou_threshold": 0.3,
                "num_videos": 2,
                "num_queries": 3,
                "Recall@10": 0.5,
                "mAP@10": 0.25,
                "elapsed_sec": 10.0,
                "queries_per_sec": 0.3,
            }
        ),
        encoding="utf-8",
    )
    summary_20_10.write_text(
        json.dumps(
            {
                "profile": "activitynet_visual_only",
                "top_k": 10,
                "window_sec": 20.0,
                "stride_sec": 10.0,
                "tiou_threshold": 0.3,
                "num_videos": 2,
                "num_queries": 3,
                "Recall@10": 0.7,
                "mAP@10": 0.4,
                "elapsed_sec": 8.0,
                "queries_per_sec": 0.375,
            }
        ),
        encoding="utf-8",
    )
    costs.write_text(
        json.dumps(
            [
                {
                    "ablation_type": "moment_windows",
                    "setting": "10s/5s",
                    "total_units": 30,
                    "relative_to_default": 1.0,
                },
                {
                    "ablation_type": "moment_windows",
                    "setting": "20s/10s",
                    "total_units": 14,
                    "relative_to_default": 0.4667,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = summarize_temporal_tradeoff(
        [summary_10_5, summary_20_10],
        cost_path=costs,
        csv_path=tmp_path / "tradeoff.csv",
        markdown_path=tmp_path / "tradeoff.md",
        latex_path=tmp_path / "tradeoff.tex",
    )

    by_setting = {row["temporal_setting"]: row for row in rows}
    assert by_setting["10s/5s"]["candidate_windows"] == 30
    assert by_setting["20s/10s"]["window_cost_relative"] == 0.4667
    assert "coarser localized moments" in (tmp_path / "tradeoff.md").read_text(
        encoding="utf-8"
    )
    assert "\\label{tab:activitynet-temporal-tradeoff}" in (
        tmp_path / "tradeoff.tex"
    ).read_text(encoding="utf-8")


def test_check_activitynet_paper_artifacts_detects_stale_metrics(tmp_path: Path):
    visual_summary = tmp_path / "visual_summary.json"
    coarse_summary = tmp_path / "coarse_summary.json"
    profile_csv = tmp_path / "profile.csv"
    temporal_csv = tmp_path / "temporal.csv"
    cost_json = tmp_path / "costs.json"
    findings = tmp_path / "findings.md"
    draft = tmp_path / "draft.md"

    visual_summary.write_text(
        json.dumps(
            {
                "profile": "activitynet_visual_only",
                "top_k": 10,
                "window_sec": 10.0,
                "stride_sec": 5.0,
                "tiou_threshold": 0.3,
                "num_videos": 2,
                "num_queries": 3,
                "Recall@10": 0.5,
                "mAP@10": 0.25,
                "elapsed_sec": 10.0,
                "queries_per_sec": 0.3,
            }
        ),
        encoding="utf-8",
    )
    coarse_summary.write_text(
        json.dumps(
            {
                "profile": "activitynet_visual_only",
                "top_k": 10,
                "window_sec": 20.0,
                "stride_sec": 10.0,
                "tiou_threshold": 0.3,
                "num_videos": 2,
                "num_queries": 3,
                "Recall@10": 0.7,
                "mAP@10": 0.4,
                "elapsed_sec": 8.0,
                "queries_per_sec": 0.375,
            }
        ),
        encoding="utf-8",
    )
    profile_csv.write_text(
        "\n".join(
            [
                "profile,num_videos,num_queries,top_k,tiou_threshold,Recall@10,mAP@10",
                "activitynet_visual_only,2,3,10,0.3,0.5,0.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    temporal_csv.write_text(
        "\n".join(
            [
                "profile,temporal_setting,num_videos,num_queries,top_k,tiou_threshold,Recall@10,mAP@10,candidate_windows,window_cost_relative,elapsed_sec,queries_per_sec",
                "activitynet_visual_only,10s/5s,2,3,10,0.3,0.5,0.25,30,1.0,10.0,0.3",
                "activitynet_visual_only,20s/10s,2,3,10,0.3,0.7,0.4,14,0.4667,8.0,0.375",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cost_json.write_text(
        json.dumps(
            [
                {
                    "ablation_type": "moment_windows",
                    "setting": "10s/5s",
                    "total_units": 30,
                    "relative_to_default": 1.0,
                },
                {
                    "ablation_type": "moment_windows",
                    "setting": "20s/10s",
                    "total_units": 14,
                    "relative_to_default": 0.4667,
                },
            ]
        ),
        encoding="utf-8",
    )
    findings.write_text(
        "controlled proxy benchmark\nCASTLE remains the downstream lifelog setting\nActivityNet results should not be described as a direct win\n",
        encoding="utf-8",
    )
    draft.write_text(
        "controlled proxy benchmark\ntemporal granularity trade-off\nDo not claim that this system outperforms WorldMM\n",
        encoding="utf-8",
    )

    errors = check_activitynet_paper_artifacts(
        visual_summary_path=visual_summary,
        coarse_summary_path=coarse_summary,
        profile_table_csv_path=profile_csv,
        temporal_tradeoff_csv_path=temporal_csv,
        cost_json_path=cost_json,
        findings_path=findings,
        report_draft_path=draft,
    )

    assert any("mAP@10 mismatch" in error for error in errors)


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
