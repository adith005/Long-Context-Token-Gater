import pytest
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_gate_heuristic(tmp_path):
    # minimal sanity check
    payload = {
        "query": "apples",
        "strategy": "heuristic",
        "top_k": 3,
        "token_budget": 50
    }
    resp = client.post("/gate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "selected" in data
    assert isinstance(data["token_estimate"], int)
    assert data["meta"]["strategy"] == "heuristic"
