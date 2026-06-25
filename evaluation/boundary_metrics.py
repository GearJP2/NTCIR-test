from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryMetrics:
    predicted_count: int
    reference_count: int
    matched_count: int
    precision: float
    recall: float
    f1: float
    mean_absolute_error_ms: float | None


def evaluate_boundaries(
    *,
    predicted_ms: list[int],
    reference_ms: list[int],
    tolerance_ms: int,
) -> BoundaryMetrics:
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")

    predicted = sorted(predicted_ms)
    reference = sorted(reference_ms)
    predicted_index = 0
    reference_index = 0
    errors: list[int] = []

    while predicted_index < len(predicted) and reference_index < len(reference):
        prediction = predicted[predicted_index]
        expected = reference[reference_index]
        difference = prediction - expected
        if abs(difference) <= tolerance_ms:
            errors.append(abs(difference))
            predicted_index += 1
            reference_index += 1
        elif prediction < expected:
            predicted_index += 1
        else:
            reference_index += 1

    matched = len(errors)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(reference) if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return BoundaryMetrics(
        predicted_count=len(predicted),
        reference_count=len(reference),
        matched_count=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_absolute_error_ms=sum(errors) / matched if matched else None,
    )
