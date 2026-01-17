from anstkit.governance.deployment_modes import (
    DeploymentMode,
    ModeController,
    ShadowResult,
)
from anstkit.schemas import ActionType, ControlAction, PlantState


def test_shadow_mode_logs_without_executing():
    controller = ModeController(mode=DeploymentMode.SHADOW)
    state = PlantState()
    actions = [ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7)]

    result = controller.process(state, actions, approved=True)

    assert isinstance(result, ShadowResult)
    assert result.would_execute
    assert not result.actually_executed


def test_guarded_mode_requires_human_approval():
    controller = ModeController(mode=DeploymentMode.GUARDED)
    state = PlantState()
    actions = [ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7)]

    result = controller.process(state, actions, approved=True, human_confirmed=False)

    assert not result.actually_executed
    assert result.awaiting_confirmation


def test_expanded_mode_auto_executes_approved():
    controller = ModeController(mode=DeploymentMode.EXPANDED)
    state = PlantState()
    actions = [ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7)]

    result = controller.process(state, actions, approved=True)

    assert result.actually_executed


def test_guarded_mode_executes_with_human_confirmation():
    controller = ModeController(mode=DeploymentMode.GUARDED)
    state = PlantState()
    actions = [ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7)]

    result = controller.process(state, actions, approved=True, human_confirmed=True)

    assert result.actually_executed
    assert not result.awaiting_confirmation


def test_expanded_mode_rejects_unapproved():
    controller = ModeController(mode=DeploymentMode.EXPANDED)
    state = PlantState()
    actions = [ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7)]

    result = controller.process(state, actions, approved=False)

    assert not result.actually_executed
    assert not result.would_execute
    assert result.rejection_reason == "Triad rejected actions"
