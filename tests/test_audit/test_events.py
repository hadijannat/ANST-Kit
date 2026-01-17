
import pytest

from anstkit.audit.events import AuditEvent, EventType
from anstkit.audit.store import AuditStore


def test_audit_event_creation():
    event = AuditEvent(
        event_type=EventType.PROPOSAL_SUBMITTED,
        session_id="sess-123",
        payload={"goal": "increase throughput"},
    )
    assert event.event_type == EventType.PROPOSAL_SUBMITTED
    assert event.timestamp is not None
    assert event.event_id is not None


def test_audit_store_append_only():
    store = AuditStore(":memory:")  # SQLite in-memory
    event = AuditEvent(
        event_type=EventType.PROPOSAL_SUBMITTED,
        session_id="sess-123",
        payload={"goal": "test"},
    )
    store.append(event)
    events = store.query(session_id="sess-123")
    assert len(events) == 1
    assert events[0].event_type == EventType.PROPOSAL_SUBMITTED


def test_audit_store_immutable():
    store = AuditStore(":memory:")
    event = AuditEvent(
        event_type=EventType.DECISION_MADE,
        session_id="sess-123",
        payload={"approved": True},
    )
    store.append(event)
    # Attempting to modify should fail
    with pytest.raises(ValueError):
        store.delete(event.event_id)


def test_audit_store_query_by_event_type():
    store = AuditStore(":memory:")
    store.append(AuditEvent(
        event_type=EventType.PROPOSAL_SUBMITTED,
        session_id="sess-1",
        payload={"goal": "test1"},
    ))
    store.append(AuditEvent(
        event_type=EventType.DECISION_MADE,
        session_id="sess-1",
        payload={"approved": True},
    ))
    store.append(AuditEvent(
        event_type=EventType.PROPOSAL_SUBMITTED,
        session_id="sess-2",
        payload={"goal": "test2"},
    ))

    proposals = store.query(event_type=EventType.PROPOSAL_SUBMITTED)
    assert len(proposals) == 2

    decisions = store.query(event_type=EventType.DECISION_MADE)
    assert len(decisions) == 1


def test_audit_event_with_parent():
    parent = AuditEvent(
        event_type=EventType.PROPOSAL_SUBMITTED,
        session_id="sess-123",
        payload={"goal": "test"},
    )
    child = AuditEvent(
        event_type=EventType.STRUCTURAL_GATE_PASS,
        session_id="sess-123",
        payload={"status": "pass"},
        parent_event_id=parent.event_id,
    )
    assert child.parent_event_id == parent.event_id
