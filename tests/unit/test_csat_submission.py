import io

import pytest

from evaluation.csat_submission import CSATResult, select_csat_results, write_csat_submission
from scripts.export_csat_results import main


def result(start=0, end=10_000, score=0.9, path="day1/Allie/video/08.mp4", **kwargs):
    return CSATResult(source_relative_path=path, start_ms=start, end_ms=end,
                      score=score, reranked=kwargs.get("reranked", True))


def test_selection_threshold_ties_adjacency_and_sources():
    first = result()
    adjacent = result(10_000, 20_000, 0.5)
    other_source = result(path="day1/Luca/video/08.mp4")
    other_day = result(path="day2/Allie/video/08.mp4")
    assert select_csat_results([
        first, result(5_000, 15_000), adjacent, other_source, other_day,
        result(20_000, 30_000, 0.49), result(30_000, 40_000, reranked=False),
    ]) == [first, other_source, other_day, adjacent]


def test_overlap_across_hour_files_uses_recording_time():
    first = result(3_590_000, 3_610_000)
    assert select_csat_results([first, result(path="day1/Allie/video/09.mp4")]) == [first]


def test_submission_format_and_limit():
    handle = io.StringIO()
    rows = [result(i * 1000, (i + 1) * 1000) for i in range(101)]
    assert write_csat_submission(handle, {"CSAT26001": rows},
                                 group_id="LLFL26", run_id="LLFL26CSAT02") == 100
    assert handle.getvalue().splitlines()[0] == (
        "LLFL26, LLFL26CSAT02, CSAT26001, 1, 08:00:00, 08:00:01, Allie, 0.900000"
    )


@pytest.mark.parametrize("overrides", [
    {"score": float("nan")}, {"score": 50}, {"start_ms": -1},
    {"end_ms": 0}, {"start_ms": 10_000}, {"start_ms": 0.5},
    {"source_relative_path": "day1/Allie/video/24.mp4"},
    {"source_relative_path": "day1/Allie/video/23.mp4", "end_ms": 3_600_001},
])
def test_invalid_results(overrides):
    fields = result().model_dump() | overrides
    with pytest.raises(ValueError):
        CSATResult(**fields)


def test_cli_preserves_existing_output_and_validates_before_creation(tmp_path):
    import json

    source = tmp_path / "results.json"
    output = tmp_path / "run.txt"
    source.write_text(json.dumps({"CSAT26001": [result().model_dump()]}))
    main(source, output, "LLFL26", "LLFL26CSAT02")
    original = output.read_text()
    with pytest.raises(FileExistsError):
        main(source, output, "LLFL26", "LLFL26CSAT02")
    assert output.read_text() == original
    source.write_text(json.dumps({"CSAT26001": [result(end=100).model_dump()]}))
    with pytest.raises(ValueError, match="zero duration"):
        main(source, tmp_path / "invalid.txt", "LLFL26", "LLFL26CSAT02")
    assert not (tmp_path / "invalid.txt").exists()
