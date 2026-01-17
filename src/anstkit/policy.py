"""Policy gate for action allowlists and change limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .schemas import ActionType, ControlAction, GateResult, PlantState, ValidationStatus

_ACTION_STATE_MAP: Dict[ActionType, str] = {
    ActionType.SET_PUMP_SPEED: "pump_speed",
    ActionType.SET_VALVE_OPENING: "valve_opening",
}


@dataclass
class PolicyConfig:
    """Configuration for policy gate enforcement.

    Defaults are permissive so existing demos continue to work. Tighten by
    enabling target allowlists and explicit approvals in production.
    """

    allowed_action_types: Set[ActionType] = field(
        default_factory=lambda: {ActionType.SET_PUMP_SPEED, ActionType.SET_VALVE_OPENING}
    )
    allowed_targets: Optional[Set[str]] = None
    enforce_action_allowlist: bool = True
    enforce_target_allowlist: bool = False
    enforce_value_bounds: bool = True
    max_delta: Optional[float] = 0.5
    require_explicit_approval: bool = False


def _value_bounds(action: ControlAction) -> tuple[float, float]:
    lower = action.min_value if action.min_value is not None else 0.0
    upper = action.max_value if action.max_value is not None else 1.0
    return float(lower), float(upper)


def policy_gate(
    actions: List[ControlAction],
    state: PlantState,
    cfg: PolicyConfig | None = None,
) -> GateResult:
    """Enforce allowlists and change limits before structural/physics gates."""
    cfg = cfg or PolicyConfig()
    reasons: List[str] = []
    evidence: List[Dict[str, object]] = []

    for action in actions:
        action_reasons: List[str] = []

        if cfg.enforce_action_allowlist and action.type not in cfg.allowed_action_types:
            action_reasons.append(
                f"Action type '{action.type.value}' not allowed by policy."
            )

        if cfg.enforce_target_allowlist and cfg.allowed_targets is not None:
            if action.target_id not in cfg.allowed_targets:
                action_reasons.append(
                    f"Target '{action.target_id}' not in policy allowlist."
                )

        if cfg.enforce_value_bounds:
            lower, upper = _value_bounds(action)
            if not (lower <= action.value <= upper):
                action_reasons.append(
                    f"Value {action.value:.3f} outside policy bounds [{lower:.3f}, {upper:.3f}]."
                )

        if cfg.max_delta is not None and action.type in _ACTION_STATE_MAP:
            field_name = _ACTION_STATE_MAP[action.type]
            current = float(getattr(state, field_name))
            if abs(action.value - current) > cfg.max_delta:
                action_reasons.append(
                    f"Change {action.value - current:+.3f} exceeds policy max_delta {cfg.max_delta:.3f}."
                )

        if cfg.require_explicit_approval and action.requires_approval:
            action_reasons.append("Action requires explicit approval.")

        if action_reasons:
            reasons.extend(action_reasons)

        evidence.append(
            {
                "action": action.model_dump(),
                "ok": not action_reasons,
                "reasons": action_reasons,
            }
        )

    return GateResult(
        status=ValidationStatus.PASS if not reasons else ValidationStatus.FAIL,
        reasons=reasons,
        metrics={"n_actions": len(actions)},
        evidence=evidence,
    )
