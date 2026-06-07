import pytest
from fastapi.testclient import TestClient


def test_health_liveness(client: TestClient):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_returns_empty_on_no_vectors(client: TestClient):
    resp = client.post(
        "/api/v1/search/",
        json={"text_query": "test query", "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert isinstance(data["results"], list)
