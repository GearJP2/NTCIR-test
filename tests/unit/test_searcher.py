import numpy as np
import pytest

from services.retrieval.scorer import reciprocal_rank_fusion


def test_rrf_single_list():
    hits = [{"segment_id": f"s{i}", "score": 1.0 / i} for i in range(1, 5)]
    result = reciprocal_rank_fusion([hits])
    assert result[0]["segment_id"] == "s1"


def test_rrf_merges_two_lists():
    list_a = [{"segment_id": "A"}, {"segment_id": "B"}, {"segment_id": "C"}]
    list_b = [{"segment_id": "C"}, {"segment_id": "A"}, {"segment_id": "D"}]
    result = reciprocal_rank_fusion([list_a, list_b])
    ids = [r["segment_id"] for r in result]
    # "A" and "C" appear in both lists and should rank highest
    assert ids.index("A") < ids.index("D")
    assert ids.index("C") < ids.index("D")


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
