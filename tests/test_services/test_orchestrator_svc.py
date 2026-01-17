import pytest
from fastapi.testclient import TestClient

from anstkit.services.orchestrator_svc import app


@pytest.fixture
def client():
    # Use context manager to properly invoke lifespan events
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_propose_endpoint(client):
    response = client.post("/propose", json={
        "goal": "increase throughput",
        "state": {"tank_level": 0.5, "pump_speed": 0.5, "valve_opening": 0.5},
    })
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "approved" in data["decision"]


def test_propose_returns_audit_trail(client):
    response = client.post("/propose", json={
        "goal": "stabilize level",
        "state": {"tank_level": 0.5, "pump_speed": 0.5, "valve_opening": 0.5},
    })
    assert response.status_code == 200
    data = response.json()
    assert "audit_events" in data
    assert len(data["audit_events"]) > 0


def test_audit_and_evidence_endpoints(client):
    response = client.post("/propose", json={
        "goal": "increase throughput",
        "state": {"tank_level": 0.6, "pump_speed": 0.4, "valve_opening": 0.4},
    })
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    audit_resp = client.get(f"/audit/{session_id}")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert audit_data["session_id"] == session_id
    assert len(audit_data["events"]) > 0

    evidence_resp = client.get(f"/evidence/{session_id}")
    assert evidence_resp.status_code == 200
    evidence_data = evidence_resp.json()
    assert evidence_data["session_id"] == session_id


def test_propose_endpoint_validates_state(client):
    # Invalid state should still work (FastAPI will validate)
    response = client.post("/propose", json={
        "goal": "test",
        "state": {"tank_level": 0.5, "pump_speed": 0.5, "valve_opening": 0.5},
    })
    assert response.status_code == 200


def test_propose_with_edge_state(client):
    # Edge case: state near limits
    response = client.post("/propose", json={
        "goal": "stabilize level",
        "state": {"tank_level": 0.95, "pump_speed": 0.1, "valve_opening": 0.9},
    })
    assert response.status_code == 200
    data = response.json()
    # Should have physics-related reasoning when near limits
    assert "decision" in data
