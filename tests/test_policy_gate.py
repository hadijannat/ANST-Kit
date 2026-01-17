from anstkit.policy import PolicyConfig, policy_gate
from anstkit.schemas import ActionType, ControlAction, PlantState, ValidationStatus


def test_policy_rejects_large_delta():
    state = PlantState(tank_level=0.5, pump_speed=0.5, valve_opening=0.5)
    cfg = PolicyConfig(max_delta=0.1)
    action = ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.9)

    result = policy_gate([action], state, cfg)
    assert result.status == ValidationStatus.FAIL
    assert any("max_delta" in reason for reason in result.reasons)


def test_policy_target_allowlist():
    state = PlantState(tank_level=0.5, pump_speed=0.5, valve_opening=0.5)
    cfg = PolicyConfig(
        allowed_targets={"P1"},
        enforce_target_allowlist=True,
    )
    action = ControlAction(type=ActionType.SET_VALVE_OPENING, target_id="V2", value=0.4)

    result = policy_gate([action], state, cfg)
    assert result.status == ValidationStatus.FAIL
    assert any("allowlist" in reason for reason in result.reasons)
