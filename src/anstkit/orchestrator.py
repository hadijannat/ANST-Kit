"""Triad orchestrator: Propose → Verify → Execute.

This is the control logic that composes:
- agent (Brain)
- structural gate (Map)
- physics gate (Guardrail)

The orchestrator is deliberately deterministic so runs can be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import uuid

from .agent_demo import DemoAgent
from .plant_graph import PlantGraph
from .physics_pinn import PhysicsGateConfig, TankPINN, apply_actions_to_state, physics_gate
from .schemas import Decision, PlantState, ValidationStatus

if TYPE_CHECKING:
    from .audit.store import AuditStore


@dataclass
class OrchestratorConfig:
    max_iterations: int = 3


class TriadOrchestrator:
    """Orchestrator for the triadic runtime assurance pattern.

    Coordinates the propose-verify-execute loop:
    1. Agent (Brain) proposes actions based on goal and state
    2. Structural gate (Map) validates topology constraints
    3. Physics gate (Guardrail) validates physical constraints via PINN
    4. If approved, actions are executed

    Optionally logs all events to an audit store for non-repudiation.
    """

    def __init__(
        self,
        agent: DemoAgent,
        plant: PlantGraph,
        pinn: TankPINN,
        state: PlantState | None = None,
        cfg: OrchestratorConfig | None = None,
        physics_cfg: PhysicsGateConfig | None = None,
        audit_store: Optional["AuditStore"] = None,
    ):
        self.agent = agent
        self.plant = plant
        self.pinn = pinn
        self.state = state or PlantState()
        self.cfg = cfg or OrchestratorConfig()
        self.physics_cfg = physics_cfg or PhysicsGateConfig()
        self.audit_store = audit_store
        self._session_id = str(uuid.uuid4())

    @property
    def session_id(self) -> str:
        """Unique session ID for this orchestrator instance."""
        return self._session_id

    def _log_event(self, event_type: str, payload: dict) -> None:
        """Log an audit event if audit store is configured."""
        if self.audit_store is None:
            return
        from .audit.events import AuditEvent, EventType

        self.audit_store.append(
            AuditEvent(
                event_type=EventType(event_type),
                session_id=self._session_id,
                payload=payload,
            )
        )

    def step(self, goal: str) -> Decision:
        """Execute one propose-verify-execute cycle.

        Args:
            goal: Natural language goal for the agent.

        Returns:
            Decision with approval status, gate results, and final actions.
        """
        proposal = self.agent.propose(goal, self.state)

        # Log proposal
        self._log_event(
            "proposal_submitted",
            {"goal": goal, "actions": [a.model_dump() for a in proposal.actions]},
        )

        last_struct = None
        last_phys = None

        for iteration in range(self.cfg.max_iterations):
            struct = self.plant.structural_gate(proposal.actions)
            last_struct = struct

            # Log structural gate result
            if struct.status == ValidationStatus.PASS:
                self._log_event(
                    "structural_gate_pass",
                    {"iteration": iteration, "reasons": struct.reasons},
                )
            else:
                self._log_event(
                    "structural_gate_fail",
                    {"iteration": iteration, "reasons": struct.reasons},
                )
                proposal = self.agent.revise(proposal, struct.reasons, self.state)
                continue

            phys = physics_gate(self.pinn, self.state, proposal.actions, self.physics_cfg)
            last_phys = phys

            # Log physics gate result
            if phys.status == ValidationStatus.PASS:
                self._log_event(
                    "physics_gate_pass",
                    {"iteration": iteration, "reasons": phys.reasons, "metrics": phys.metrics},
                )
            else:
                self._log_event(
                    "physics_gate_fail",
                    {"iteration": iteration, "reasons": phys.reasons, "metrics": phys.metrics},
                )
                proposal = self.agent.revise(proposal, phys.reasons, self.state)
                continue

            # Both gates passed → execute (demo state transition)
            self.state = apply_actions_to_state(self.state, proposal.actions)

            decision = Decision(
                approved=True,
                structural=struct,
                physics=phys,
                final_actions=proposal.actions,
            )

            # Log decision
            self._log_event(
                "decision_made",
                {"approved": True, "actions": [a.model_dump() for a in proposal.actions]},
            )

            return decision

        # Failed to converge to a safe plan
        if last_phys is None:
            # If physics never ran, set a default
            from .schemas import GateResult

            last_phys = GateResult(
                status=ValidationStatus.FAIL,
                reasons=["Physics gate not reached."],
                metrics={},
            )

        decision = Decision(
            approved=False,
            structural=last_struct,
            physics=last_phys,
            final_actions=[],
        )

        # Log rejection
        self._log_event(
            "decision_made",
            {"approved": False, "reason": "Failed to converge to safe plan"},
        )

        return decision
