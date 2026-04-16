"""Integration tests for FastAPI endpoints (requires running infrastructure)."""

import pytest


@pytest.mark.integration
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data


@pytest.mark.integration
def test_ingest(client):
    payload = {
        "events": [
            {"id": "test-1", "timestamp": "2026-01-01T00:00:00", "payload": {"x": 1.0}}
        ]
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] == 1
    assert data["topic"] == "raw-events"
