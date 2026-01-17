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

from .schemas import ActionType, ControlAction, Proposal


@dataclass
class DemoAgentConfig:
    p_hallucinate_id: float = 0.20
    p_out_of_range: float = 0.20


class DemoAgent:
    def __init__(self, seed: int = 7, cfg: DemoAgentConfig | None = None):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg or DemoAgentConfig()

    def propose(self, goal: str, state) -> Proposal:
        actions: List[ControlAction] = []

        # Default targets
        pump_id = "P1"
        valve_id = "V2"

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

    def revise(self, proposal: Proposal, feedback: List[str], state) -> Proposal:
        """Very small "self-correction" loop based on validator feedback."""

        new_actions = []
        for a in proposal.actions:
            new_a = a.model_copy(deep=True)

            if "Unknown target_id" in " ".join(feedback):
                # Replace unknown valves with V2 in demo
                if new_a.type == ActionType.SET_VALVE_OPENING:
                    new_a.target_id = "V2"

            if "out of allowed range" in " ".join(feedback):
                new_a.value = float(np.clip(new_a.value, 0.0, 1.0))

            if "exceeds limit" in " ".join(feedback):
                # If physics says overflow risk, reduce pump a bit.
                if new_a.type == ActionType.SET_PUMP_SPEED:
                    new_a.value = float(np.clip(state.pump_speed - 0.1, 0.0, 1.0))

            new_actions.append(new_a)

        return Proposal(goal=proposal.goal, actions=new_actions, rationale=proposal.rationale, confidence=proposal.confidence)
