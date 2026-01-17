"""FastAPI orchestrator service with audit integration."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from anstkit.agent_demo import DemoAgent
from anstkit.audit.events import AuditEvent, EventType
from anstkit.audit.store import AuditStore
from anstkit.orchestrator import TriadOrchestrator
from anstkit.physics_pinn import PhysicsGateConfig, TankPINN, load_pinn, train_pinn
from anstkit.plant_graph import PlantGraph
from anstkit.schemas import PlantState

from .schemas import (
    ActionResponse,
    AuditEventResponse,
    DecisionResponse,
    HealthResponse,
    ProposeRequest,
    ProposeResponse,
)

# Global components (initialized in lifespan)
agent: Optional[DemoAgent] = None
plant: Optional[PlantGraph] = None
pinn: Optional[TankPINN] = None
audit_store: Optional[AuditStore] = None

DEFAULT_WEIGHTS = Path(__file__).parent.parent.parent.parent / "models" / "tank_pinn.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global agent, plant, pinn, audit_store

    agent = DemoAgent(seed=42)
    plant = PlantGraph()

    # Train PINN if weights don't exist
    if not DEFAULT_WEIGHTS.exists():
        DEFAULT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        train_pinn(DEFAULT_WEIGHTS, steps=1500, seed=42)

    pinn = load_pinn(DEFAULT_WEIGHTS)
    audit_store = AuditStore(":memory:")  # In-memory for demo

    yield

    # Cleanup
    if audit_store:
        audit_store.close()


app = FastAPI(
    title="ANST-Kit Orchestrator Service",
    version="0.1.0",
    description="Triad Runtime Assurance API: Agent + Structural Gate + Physics Gate",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="orchestrator")


@app.post("/propose", response_model=ProposeResponse)
def propose(request: ProposeRequest):
    """Submit a proposal for the triad to evaluate.

    The agent proposes actions based on the goal and current state.
    The structural gate validates topology constraints.
    The physics gate validates physics constraints using PINN.

    Returns the decision with an audit trail of events.
    """
    session_id = str(uuid.uuid4())

    # Log proposal submitted
    audit_store.append(
        AuditEvent(
            event_type=EventType.PROPOSAL_SUBMITTED,
            session_id=session_id,
            payload={"goal": request.goal},
        )
    )

    # Convert request state to PlantState
    state = PlantState(
        tank_level=request.state.tank_level,
        pump_speed=request.state.pump_speed,
        valve_opening=request.state.valve_opening,
    )

    # Create orchestrator and step
    physics_cfg = PhysicsGateConfig(horizon=2.0, n_eval=32)
    orch = TriadOrchestrator(
        agent=agent,
        plant=plant,
        pinn=pinn,
        state=state,
        physics_cfg=physics_cfg,
    )

    decision = orch.step(request.goal)

    # Log gate results
    if decision.structural.status.value == "pass":
        audit_store.append(
            AuditEvent(
                event_type=EventType.STRUCTURAL_GATE_PASS,
                session_id=session_id,
                payload={"reasons": decision.structural.reasons},
            )
        )
    else:
        audit_store.append(
            AuditEvent(
                event_type=EventType.STRUCTURAL_GATE_FAIL,
                session_id=session_id,
                payload={"reasons": decision.structural.reasons},
            )
        )

    if decision.physics.status.value == "pass":
        audit_store.append(
            AuditEvent(
                event_type=EventType.PHYSICS_GATE_PASS,
                session_id=session_id,
                payload={"reasons": decision.physics.reasons},
            )
        )
    else:
        audit_store.append(
            AuditEvent(
                event_type=EventType.PHYSICS_GATE_FAIL,
                session_id=session_id,
                payload={"reasons": decision.physics.reasons},
            )
        )

    # Log final decision
    audit_store.append(
        AuditEvent(
            event_type=EventType.DECISION_MADE,
            session_id=session_id,
            payload={"approved": decision.approved},
        )
    )

    # Query all events for this session
    events = audit_store.query(session_id=session_id)

    # Build response
    return ProposeResponse(
        decision=DecisionResponse(
            approved=decision.approved,
            structural_status=decision.structural.status.value,
            physics_status=decision.physics.status.value,
            actions=[
                ActionResponse(
                    type=a.type.value,
                    target_id=a.target_id,
                    value=a.value,
                )
                for a in decision.final_actions
            ],
            reasons=decision.structural.reasons + decision.physics.reasons,
        ),
        audit_events=[
            AuditEventResponse(
                event_id=e.event_id,
                event_type=e.event_type.value,
                timestamp=e.timestamp.isoformat(),
            )
            for e in events
        ],
        session_id=session_id,
    )
