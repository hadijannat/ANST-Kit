from pathlib import Path

import pytest

from anstkit.agent_demo import DemoAgent
from anstkit.audit.events import EventType
from anstkit.audit.store import AuditStore
from anstkit.orchestrator import TriadOrchestrator
from anstkit.physics_pinn import load_pinn, train_pinn
from anstkit.plant_graph import PlantGraph
from anstkit.schemas import PlantState


@pytest.fixture
def pinn():
    """Get or create PINN model for tests."""
    weights = Path(__file__).parent.parent / "models" / "tank_pinn.pt"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        train_pinn(weights, steps=500, seed=42)
    return load_pinn(weights)


def test_orchestrator_emits_audit_events(pinn):
    store = AuditStore(":memory:")
    agent = DemoAgent(seed=42)
    plant = PlantGraph()

    orch = TriadOrchestrator(
        agent=agent,
        plant=plant,
        pinn=pinn,
        state=PlantState(),
        audit_store=store,
    )

    orch.step("increase throughput")

    events = store.query(session_id=orch.session_id)
    event_types = {e.event_type for e in events}

    assert EventType.PROPOSAL_SUBMITTED in event_types
    assert EventType.DECISION_MADE in event_types


def test_orchestrator_logs_gate_results(pinn):
    store = AuditStore(":memory:")
    agent = DemoAgent(seed=42)
    plant = PlantGraph()

    orch = TriadOrchestrator(
        agent=agent,
        plant=plant,
        pinn=pinn,
        state=PlantState(),
        audit_store=store,
    )

    orch.step("stabilize level")
    events = store.query(session_id=orch.session_id)
    event_types = {e.event_type for e in events}

    # Should have gate results logged
    has_structural = (
        EventType.STRUCTURAL_GATE_PASS in event_types
        or EventType.STRUCTURAL_GATE_FAIL in event_types
    )
    has_physics = (
        EventType.PHYSICS_GATE_PASS in event_types
        or EventType.PHYSICS_GATE_FAIL in event_types
    )

    assert has_structural
    assert has_physics


def test_orchestrator_without_audit_still_works(pinn):
    """Audit store is optional - orchestrator should work without it."""
    agent = DemoAgent(seed=42)
    plant = PlantGraph()

    orch = TriadOrchestrator(
        agent=agent,
        plant=plant,
        pinn=pinn,
        state=PlantState(),
        # No audit_store
    )

    decision = orch.step("increase throughput")
    assert isinstance(decision.approved, bool)


def test_orchestrator_session_id_is_unique(pinn):
    store = AuditStore(":memory:")
    agent = DemoAgent(seed=42)
    plant = PlantGraph()

    orch1 = TriadOrchestrator(
        agent=agent,
        plant=plant,
        pinn=pinn,
        state=PlantState(),
        audit_store=store,
    )
    orch2 = TriadOrchestrator(
        agent=agent,
        plant=plant,
        pinn=pinn,
        state=PlantState(),
        audit_store=store,
    )

    assert orch1.session_id != orch2.session_id
