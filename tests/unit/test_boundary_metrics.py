from evaluation.boundary_metrics import evaluate_boundaries


def test_evaluate_boundaries_matches_each_reference_at_most_once():
    result = evaluate_boundaries(
        predicted_ms=[9_000, 10_000, 31_000],
        reference_ms=[10_000, 30_000],
        tolerance_ms=2_000,
    )

    assert result.matched_count == 2
    assert result.precision == 2 / 3
    assert result.recall == 1.0
    assert result.f1 == 0.8
    assert result.mean_absolute_error_ms == 1_000
