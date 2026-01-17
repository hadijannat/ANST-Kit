from pathlib import Path

import pytest

from anstkit.physics_pinn import load_ensemble, load_pinn, physics_gate
from anstkit.schemas import ActionType, ControlAction, PlantState


def test_physics_gate_vetoes_overflow():
    weights = Path(__file__).resolve().parents[1] / "models" / "tank_pinn.pt"
    model = load_pinn(weights)

    state = PlantState(tank_level=0.9, pump_speed=0.5, valve_opening=0.5)
    actions = [
        ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=1.0),
        ControlAction(type=ActionType.SET_VALVE_OPENING, target_id="V2", value=0.0),
    ]

    gate = physics_gate(model, state, actions)
    assert gate.status.value == "fail"
    assert any("exceeds limit" in r for r in gate.reasons)


def test_physics_gate_ensemble_uncertainty_metrics():
    weights = Path(__file__).resolve().parents[1] / "models" / "tank_pinn.pt"
    ensemble = load_ensemble([weights, weights])

    state = PlantState(tank_level=0.6, pump_speed=0.4, valve_opening=0.4)
    actions = [
        ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.6),
        ControlAction(type=ActionType.SET_VALVE_OPENING, target_id="V2", value=0.4),
    ]

    gate = physics_gate(ensemble, state, actions)
    assert "uncertainty_mean" in gate.metrics
    assert gate.metrics["uncertainty_mean"] == pytest.approx(0.0, abs=1e-8)
