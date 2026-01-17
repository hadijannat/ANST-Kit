"""Audit event definitions for triad runtime assurance logging."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class EventType(str, Enum):
    """Types of audit events in the triad system."""

    # Proposal lifecycle
    PROPOSAL_SUBMITTED = "proposal_submitted"
    PROPOSAL_MODIFIED = "proposal_modified"

    # Gate results
    STRUCTURAL_GATE_PASS = "structural_gate_pass"
    STRUCTURAL_GATE_FAIL = "structural_gate_fail"
    PHYSICS_GATE_PASS = "physics_gate_pass"
    PHYSICS_GATE_FAIL = "physics_gate_fail"

    # Decisions
    DECISION_MADE = "decision_made"
    ACTION_EXECUTED = "action_executed"
    ACTION_REJECTED = "action_rejected"

    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Immutable audit event for triad system logging.

    Attributes:
        event_type: The type of event being logged.
        session_id: Unique identifier for the session/transaction.
        payload: Arbitrary event data as key-value pairs.
        event_id: Auto-generated UUID for this event.
        timestamp: UTC timestamp of event creation.
        parent_event_id: Optional link to a parent event for causality chains.
    """

    event_type: EventType
    session_id: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parent_event_id: Optional[str] = None
