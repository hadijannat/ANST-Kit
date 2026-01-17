"""A deterministic demo agent.

This agent is deliberately imperfect:
- It sometimes proposes **non-existent actuators** (structural hallucination).
- It sometimes proposes **out-of-range setpoints** (functional hallucination).

The purpose is to give the evaluation harness something realistic to catch.

Replace this module with a real LLM agent in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import get_settings
from .schemas import ActionType, ControlAction, PlantState, Proposal


@dataclass
class DemoAgentConfig:
    """Configuration for demo agent behavior.

    Defaults are loaded from centralized config if not specified.
    """

    p_hallucinate_id: Optional[float] = None
    p_out_of_range: Optional[float] = None

    def __post_init__(self):
        """Load defaults from settings if not specified."""
        settings = get_settings()
        if self.p_hallucinate_id is None:
            self.p_hallucinate_id = settings.agent.p_hallucinate_id
        if self.p_out_of_range is None:
            self.p_out_of_range = settings.agent.p_out_of_range


class DemoAgent:
    """Deterministic demo agent for testing the triadic runtime assurance.

    This agent intentionally produces some invalid proposals to test
    the structural and physics gates.
    """

    def __init__(self, seed: Optional[int] = None, cfg: Optional[DemoAgentConfig] = None):
        """Initialize the demo agent.

        Args:
            seed: Random seed for reproducibility. If None, uses config default.
            cfg: Agent configuration. If None, loads from centralized config.
        """
        settings = get_settings()
        seed = seed if seed is not None else settings.seed
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg or DemoAgentConfig()

    def propose(self, goal: str, state: PlantState) -> Proposal:
        """Propose actions based on the goal and current state.

        Args:
            goal: Natural language goal (e.g., "increase throughput").
            state: Current plant state.

        Returns:
            Proposal with suggested actions.
        """
        settings = get_settings()
        actions: List[ControlAction] = []

        # Default targets from config
        pump_id = settings.plant.pump_id
        valve_id = settings.plant.outlet_valve_id

        # Hallucinate an ID occasionally
        if self.rng.random() < self.cfg.p_hallucinate_id:
            valve_id = "V99"  # does not exist in the demo graph

        # Simple policy: if goal includes 'increase', raise pump and open valve a bit
        if "increase" in goal.lower() or "throughput" in goal.lower() or "flow" in goal.lower():
            pump = state.pump_speed + float(self.rng.normal(0.15, 0.05))
            valve = state.valve_opening + float(self.rng.normal(0.10, 0.05))
        elif "decrease" in goal.lower() or "reduce" in goal.lower() or "stabilize" in goal.lower():
            pump = state.pump_speed - float(self.rng.normal(0.10, 0.05))
            valve = state.valve_opening + float(self.rng.normal(0.05, 0.05))
        else:
            pump = state.pump_speed + float(self.rng.normal(0.0, 0.10))
            valve = state.valve_opening + float(self.rng.normal(0.0, 0.10))

        # Out-of-range occasionally
        if self.rng.random() < self.cfg.p_out_of_range:
            pump = pump + 0.8

        actions.append(ControlAction(type=ActionType.SET_PUMP_SPEED, target_id=pump_id, value=float(pump)))
        actions.append(ControlAction(type=ActionType.SET_VALVE_OPENING, target_id=valve_id, value=float(valve)))

        return Proposal(goal=goal, actions=actions, rationale="Demo policy", confidence=None)

    def revise(
        self,
        proposal: Proposal,
        feedback: List[str],
        state: PlantState,
    ) -> Proposal:
        """Revise a proposal based on validator feedback.

        Args:
            proposal: Previous proposal that failed validation.
            feedback: List of rejection reasons from gates.
            state: Current plant state.

        Returns:
            Revised proposal with corrected actions.
        """
        settings = get_settings()
        new_actions: List[ControlAction] = []

        for a in proposal.actions:
            new_a = a.model_copy(deep=True)

            if "Unknown target_id" in " ".join(feedback):
                # Replace unknown valves with valid outlet valve from config
                if new_a.type == ActionType.SET_VALVE_OPENING:
                    new_a.target_id = settings.plant.outlet_valve_id

            if "out of allowed range" in " ".join(feedback):
                new_a.value = float(np.clip(new_a.value, 0.0, 1.0))

            if "exceeds limit" in " ".join(feedback):
                # If physics says overflow risk, reduce pump a bit.
                if new_a.type == ActionType.SET_PUMP_SPEED:
                    new_a.value = float(np.clip(state.pump_speed - 0.1, 0.0, 1.0))

            new_actions.append(new_a)

        return Proposal(
            goal=proposal.goal,
            actions=new_actions,
            rationale=proposal.rationale,
            confidence=proposal.confidence,
        )
