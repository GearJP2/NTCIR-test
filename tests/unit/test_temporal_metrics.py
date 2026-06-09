from evaluation.manifest import EvaluationQuery, GroundTruthMoment
from evaluation.temporal_metrics import (
    RetrievedMoment,
    average_precision_at_k,
    mean_average_precision_at_k,
    recall_at_k,
    temporal_iou,
)


def _query(
    query_id: str = "q1",
    media_id: str = "v_123",
    start_sec: float = 40.0,
    end_sec: float = 50.0,
) -> EvaluationQuery:
    return EvaluationQuery(
        query_id=query_id,
        media_id=media_id,
        query="woman doing sit ups",
        ground_truth=GroundTruthMoment(start_sec=start_sec, end_sec=end_sec),
    )


def test_temporal_iou_for_overlapping_intervals():
    gt = GroundTruthMoment(start_sec=40.0, end_sec=50.0)
    result = RetrievedMoment(media_id="v_123", start_sec=45.0, end_sec=55.0, score=0.9)

    assert temporal_iou(gt, result) == 5.0 / 15.0


def test_temporal_iou_for_disjoint_intervals():
    gt = GroundTruthMoment(start_sec=40.0, end_sec=50.0)
    result = RetrievedMoment(media_id="v_123", start_sec=60.0, end_sec=70.0, score=0.9)

    assert temporal_iou(gt, result) == 0.0


def test_recall_at_k_counts_temporal_match_in_top_k():
    query = _query()
    results = {
        "q1": [
            RetrievedMoment(media_id="v_123", start_sec=0.0, end_sec=10.0, score=0.9),
            RetrievedMoment(media_id="v_123", start_sec=45.0, end_sec=55.0, score=0.8),
        ]
    }

    assert recall_at_k([query], results, k=10, tiou_threshold=0.3) == 1.0


def test_recall_at_k_rejects_wrong_media_id():
    query = _query()
    results = {
        "q1": [
            RetrievedMoment(media_id="v_other", start_sec=40.0, end_sec=50.0, score=0.9),
        ]
    }

    assert recall_at_k([query], results, k=10, tiou_threshold=0.3) == 0.0


def test_recall_at_k_respects_k():
    query = _query()
    results = {
        "q1": [
            RetrievedMoment(media_id="v_123", start_sec=0.0, end_sec=10.0, score=0.9),
            RetrievedMoment(media_id="v_123", start_sec=40.0, end_sec=50.0, score=0.8),
        ]
    }

    assert recall_at_k([query], results, k=1, tiou_threshold=0.3) == 0.0


def test_average_precision_at_k_for_single_ground_truth_query():
    query = _query()
    ranked = [
        RetrievedMoment(media_id="v_123", start_sec=0.0, end_sec=10.0, score=0.9),
        RetrievedMoment(media_id="v_123", start_sec=40.0, end_sec=50.0, score=0.8),
    ]

    assert average_precision_at_k(query, ranked, k=10, tiou_threshold=0.3) == 0.5


def test_mean_average_precision_at_k():
    q1 = _query(query_id="q1")
    q2 = _query(query_id="q2", start_sec=80.0, end_sec=90.0)
    results = {
        "q1": [
            RetrievedMoment(media_id="v_123", start_sec=40.0, end_sec=50.0, score=0.8),
        ],
        "q2": [
            RetrievedMoment(media_id="v_123", start_sec=0.0, end_sec=10.0, score=0.9),
        ],
    }

    assert mean_average_precision_at_k([q1, q2], results, k=10, tiou_threshold=0.3) == 0.5
