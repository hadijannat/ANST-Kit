from anstkit.plant_graph import PlantGraph
from anstkit.schemas import ActionType, ControlAction


def test_unknown_target_rejected():
    plant = PlantGraph()
    action = ControlAction(type=ActionType.SET_VALVE_OPENING, target_id="V99", value=0.5)
    ok, reasons = plant.validate_action(action)
    assert not ok
    assert any("Unknown target_id" in r for r in reasons)
