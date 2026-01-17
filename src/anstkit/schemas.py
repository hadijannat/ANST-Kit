"""Typed schemas for ANST-Kit.

These models enforce *machine-parseable* actions and decisions so the
"Brain" (agent) cannot smuggle ambiguous commands into the control path.

In an industrial-grade implementation, extend these schemas with:
- Units of measure
- Asset taxonomy / ontology alignment (e.g., DEXPI)
- Role-based access control (RBAC)
- Provenance + cryptographic signing for non-repudiation
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    SET_PUMP_SPEED = "set_pump_speed"        # normalized 0..1 for demo
    SET_VALVE_OPENING = "set_valve_opening"  # normalized 0..1 for demo


class ControlAction(BaseModel):
    """A single actuator command.

    For safety-critical deployments, this should be the *only* interface
    from the agent to the execution system.
    """

    type: ActionType
    target_id: str
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Proposal(BaseModel):
    """An agent proposal: one or more actions plus optional rationale."""

    goal: str
    actions: List[ControlAction]
    rationale: str = ""
    confidence: Optional[float] = None


class PlantState(BaseModel):
    """Minimal plant state for the demo.

    - tank_level is normalized (0..1)
    - pump_speed and valve_opening are normalized (0..1)
    """

    tank_level: float = 0.5
    pump_speed: float = 0.5
    valve_opening: float = 0.5


class ValidationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class GateResult(BaseModel):
    """Result of one validation gate (structural or physics)."""

    status: ValidationStatus
    reasons: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """Combined decision after both gates."""

    approved: bool
    structural: GateResult
    physics: GateResult
    final_actions: List[ControlAction] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    n: int
    baseline_unsafe_rate: float
    triad_unsafe_rate: float
    triad_approval_rate: float
