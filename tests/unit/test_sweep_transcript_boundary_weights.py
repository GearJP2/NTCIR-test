from scripts.sweep_transcript_boundary_weights import (
    _load_references,
    _summarize,
    _validate_weights,
)


def test_load_references_supports_negative_control(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text(
        '{"video_id":"v","tolerance_ms":10000,"description":"none"}\n',
        encoding="utf-8",
    )

    assert _load_references(path) == {
        "timestamps_ms": [],
        "tolerance_ms": 10_000,
    }


def test_validate_weights_deduplicates_while_preserving_order():
    assert _validate_weights([0.0, 0.25, 0.25, 0.5]) == [0.0, 0.25, 0.5]


def test_summarize_micro_averages_boundary_counts():
    rows = [
        {
            "transcript_weight": 0.0,
            "predicted_count": 2,
            "reference_count": 2,
            "matched_count": 1,
            "f1": 0.5,
            "event_count": 3,
            "mean_duration_ms": 10_000,
            "retrieval_recall_at_1": 0.5,
            "retrieval_recall_at_3": 1.0,
            "retrieval_recall_at_10": 1.0,
            "retrieval_mrr": 0.75,
            "retrieval_mean_best_tiou": 0.6,
            "retrieval_mean_top1_tiou": 0.4,
        },
        {
            "transcript_weight": 0.0,
            "predicted_count": 1,
            "reference_count": 1,
            "matched_count": 1,
            "f1": 1.0,
            "event_count": 2,
            "mean_duration_ms": 15_000,
            "retrieval_recall_at_1": 1.0,
            "retrieval_recall_at_3": 1.0,
            "retrieval_recall_at_10": 1.0,
            "retrieval_mrr": 1.0,
            "retrieval_mean_best_tiou": 0.8,
            "retrieval_mean_top1_tiou": 0.8,
        },
    ]

    summary = _summarize(rows, [0.0])[0]

    assert summary["boundary_precision_micro"] == 2 / 3
    assert summary["boundary_recall_micro"] == 2 / 3
    assert summary["boundary_f1_macro"] == 0.75
    assert summary["mean_event_count"] == 2.5
