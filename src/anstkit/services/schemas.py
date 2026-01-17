"""API request/response schemas for the orchestrator service."""

from typing import List

from pydantic import BaseModel, Field


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
    policy_status: str | None = None
    structural_status: str
    physics_status: str
    actions: List[ActionResponse]
    reasons: List[str]
    policy_evidence: List[dict] = Field(default_factory=list)
    structural_evidence: List[dict] = Field(default_factory=list)
    physics_evidence: List[dict] = Field(default_factory=list)


class AuditEventResponse(BaseModel):
    """Audit event in the response."""

    event_id: str
    event_type: str
    timestamp: str


class AuditEventDetailResponse(BaseModel):
    """Detailed audit event payload."""

    event_id: str
    event_type: str
    timestamp: str
    payload: dict
    parent_event_id: str | None = None


class ProposeResponse(BaseModel):
    """Full response from the propose endpoint."""

    decision: DecisionResponse
    audit_events: List[AuditEventResponse]
    session_id: str


class AuditQueryResponse(BaseModel):
    """Audit query response for a session."""

    session_id: str
    events: List[AuditEventDetailResponse]


class EvidenceResponse(BaseModel):
    """Evidence export for a session."""

    session_id: str
    policy_evidence: List[dict] = Field(default_factory=list)
    structural_evidence: List[dict] = Field(default_factory=list)
    physics_evidence: List[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
