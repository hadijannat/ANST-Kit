"""API request/response schemas for the orchestrator service."""

from pydantic import BaseModel
from typing import List, Dict, Any


class StateRequest(BaseModel):
    """Plant state input for API requests."""

    tank_level: float
    pump_speed: float
    valve_opening: float


class ProposeRequest(BaseModel):
    """Request to propose actions for a goal."""

    goal: str
    state: StateRequest


class ActionResponse(BaseModel):
    """Single action in the response."""

    type: str
    target_id: str
    value: float


class DecisionResponse(BaseModel):
    """Decision result with gate statuses."""

    approved: bool
    structural_status: str
    physics_status: str
    actions: List[ActionResponse]
    reasons: List[str]


class AuditEventResponse(BaseModel):
    """Audit event in the response."""

    event_id: str
    event_type: str
    timestamp: str


class ProposeResponse(BaseModel):
    """Full response from the propose endpoint."""

    decision: DecisionResponse
    audit_events: List[AuditEventResponse]
    session_id: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
