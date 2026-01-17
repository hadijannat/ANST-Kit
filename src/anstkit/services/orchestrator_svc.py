"""FastAPI orchestrator service with audit integration and security hardening.

This module provides REST API endpoints for the triadic runtime assurance system.
All endpoints support optional API key authentication when configured.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import ValidationError

from anstkit.agent_demo import DemoAgent, DemoAgentConfig
from anstkit.audit.events import EventType
from anstkit.audit.store import AuditStore
from anstkit.config import configure_logging, get_settings
from anstkit.orchestrator import TriadOrchestrator
from anstkit.physics_pinn import PhysicsGateConfig, TankPINN, load_pinn, train_pinn
from anstkit.plant_graph import PlantGraph
from anstkit.policy import PolicyConfig
from anstkit.schemas import PlantState

from .schemas import (
    ActionResponse,
    AuditEventDetailResponse,
    AuditEventResponse,
    AuditQueryResponse,
    DecisionResponse,
    ErrorResponse,
    EvidenceResponse,
    HealthResponse,
    ProposeRequest,
    ProposeResponse,
)

logger = logging.getLogger(__name__)

# Global components (initialized in lifespan)
agent: Optional[DemoAgent] = None
plant: Optional[PlantGraph] = None
pinn: Optional[TankPINN] = None
audit_store: Optional[AuditStore] = None

# API key security (optional based on config)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """Verify API key if authentication is required.

    Returns the API key if valid, or None if auth is disabled.
    Raises HTTPException 401 if auth is required but key is invalid/missing.
    """
    settings = get_settings()

    # If auth is not required, allow all requests
    if not settings.api.require_auth:
        return None

    # Auth is required - validate the key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key != settings.api.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global agent, plant, pinn, audit_store

    settings = get_settings()
    configure_logging(settings)

    logger.info("Initializing ANST-Kit orchestrator service...")

    # Initialize agent with config
    agent = DemoAgent(
        seed=settings.seed,
        cfg=DemoAgentConfig(
            p_hallucinate_id=settings.agent.p_hallucinate_id,
            p_out_of_range=settings.agent.p_out_of_range,
        ),
    )
    plant = PlantGraph()

    # Train PINN if weights don't exist
    weights_path = settings.pinn_weights_path
    if not weights_path.exists():
        logger.info(f"Training PINN (weights not found at {weights_path})...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        train_pinn(weights_path, steps=settings.pinn.training_steps, seed=settings.seed)

    pinn = load_pinn(weights_path)
    logger.info(f"PINN loaded from {weights_path}")

    # Initialize audit store with configured path
    audit_store = AuditStore(settings.audit.db_path)
    logger.info(f"Audit store initialized: {settings.audit.db_path}")

    logger.info("ANST-Kit orchestrator service ready")

    yield

    # Cleanup
    if audit_store:
        audit_store.close()
        logger.info("Audit store closed")


app = FastAPI(
    title="ANST-Kit Orchestrator Service",
    version="0.1.0",
    description="Triad Runtime Assurance API: Agent + Structural Gate + Physics Gate",
    lifespan=lifespan,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint (no auth required)."""
    return HealthResponse(status="healthy", service="orchestrator")


@app.post(
    "/propose",
    response_model=ProposeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
    },
)
def propose(
    request: ProposeRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Submit a proposal for the triad to evaluate.

    The agent proposes actions based on the goal and current state.
    The structural gate validates topology constraints.
    The physics gate validates physics constraints using PINN.

    Returns the decision with an audit trail of events.
    """
    try:
        settings = get_settings()

        # Validate state bounds
        if not (0.0 <= request.state.tank_level <= 1.5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"tank_level must be between 0.0 and 1.5, got {request.state.tank_level}",
            )
        if not (0.0 <= request.state.pump_speed <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"pump_speed must be between 0.0 and 1.0, got {request.state.pump_speed}",
            )
        if not (0.0 <= request.state.valve_opening <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"valve_opening must be between 0.0 and 1.0, got {request.state.valve_opening}",
            )

        # Convert request state to PlantState
        state = PlantState(
            tank_level=request.state.tank_level,
            pump_speed=request.state.pump_speed,
            valve_opening=request.state.valve_opening,
        )

        # Create orchestrator with config-based physics parameters
        physics_cfg = PhysicsGateConfig(
            horizon=settings.physics.horizon,
            n_eval=settings.physics.n_eval,
            h_max=settings.physics.h_max,
            residual_threshold=settings.physics.residual_threshold,
        )
        policy_cfg = PolicyConfig(max_delta=settings.policy.max_delta)

        orch = TriadOrchestrator(
            agent=agent,
            plant=plant,
            pinn=pinn,
            state=state,
            physics_cfg=physics_cfg,
            policy_cfg=policy_cfg,
            audit_store=audit_store,
        )

        decision = orch.step(request.goal)

        # Query all events for this session
        events = audit_store.query(session_id=orch.session_id)

        # Build response
        return ProposeResponse(
            decision=DecisionResponse(
                approved=decision.approved,
                policy_status=decision.policy.status.value if decision.policy else None,
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
                reasons=(
                    (decision.policy.reasons if decision.policy else [])
                    + decision.structural.reasons
                    + decision.physics.reasons
                ),
                policy_evidence=decision.policy.evidence if decision.policy else [],
                structural_evidence=decision.structural.evidence,
                physics_evidence=decision.physics.evidence,
            ),
            audit_events=[
                AuditEventResponse(
                    event_id=e.event_id,
                    event_type=e.event_type.value,
                    timestamp=e.timestamp.isoformat(),
                )
                for e in events
            ],
            session_id=orch.session_id,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except ValidationError as e:
        logger.warning(f"Validation error in /propose: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {e}",
        )
    except Exception:
        logger.exception("Unexpected error in /propose")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing proposal",
        )


@app.get(
    "/audit/{session_id}",
    response_model=AuditQueryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
def audit_session(
    session_id: str,
    limit: int = 1000,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Return detailed audit events for a session."""
    try:
        # Validate limit
        if limit < 1 or limit > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be between 1 and 10000",
            )

        events = audit_store.query(session_id=session_id, limit=limit)

        if not events:
            logger.info(f"No events found for session: {session_id}")
            # Return empty response rather than 404 - session may exist but have no events yet

        return AuditQueryResponse(
            session_id=session_id,
            events=[
                AuditEventDetailResponse(
                    event_id=e.event_id,
                    event_type=e.event_type.value,
                    timestamp=e.timestamp.isoformat(),
                    payload=e.payload,
                    parent_event_id=e.parent_event_id,
                )
                for e in events
            ],
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Error querying audit session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error querying audit events",
        )


@app.get(
    "/evidence/{session_id}",
    response_model=EvidenceResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
def evidence_session(
    session_id: str,
    limit: int = 1000,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Return aggregated gate evidence for a session."""
    try:
        # Validate limit
        if limit < 1 or limit > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be between 1 and 10000",
            )

        events = audit_store.query(session_id=session_id, limit=limit)
        policy_evidence = []
        structural_evidence = []
        physics_evidence = []

        for event in events:
            if event.event_type in {EventType.POLICY_GATE_PASS, EventType.POLICY_GATE_FAIL}:
                policy_evidence.extend(event.payload.get("evidence", []))
            if event.event_type in {
                EventType.STRUCTURAL_GATE_PASS,
                EventType.STRUCTURAL_GATE_FAIL,
            }:
                structural_evidence.extend(event.payload.get("evidence", []))
            if event.event_type in {EventType.PHYSICS_GATE_PASS, EventType.PHYSICS_GATE_FAIL}:
                physics_evidence.extend(event.payload.get("evidence", []))

        return EvidenceResponse(
            session_id=session_id,
            policy_evidence=policy_evidence,
            structural_evidence=structural_evidence,
            physics_evidence=physics_evidence,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Error querying evidence for session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error querying evidence",
        )
