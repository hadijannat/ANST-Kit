from anstkit.ingestion.dexpi_parser import DEXPIParser
from anstkit.plant_graph import GraphPlant
from anstkit.graph.networkx_backend import NetworkXBackend
from anstkit.schemas import ActionType, ControlAction, ValidationStatus


def test_graphplant_from_dexpi_structural_gate():
    xml = """
    <PlantModel ID="DemoPlant">
      <Equipment ID="P1" TagName="P1" ComponentClass="Pump" />
      <Equipment ID="T1" TagName="T1" ComponentClass="Tank" />
      <Equipment ID="V2" TagName="V2" ComponentClass="Valve" />
      <PipingNetworkSegment ID="S1" FromID="P1" ToID="T1" />
      <PipingNetworkSegment ID="S2" FromID="T1" ToID="V2" />
    </PlantModel>
    """
    parser = DEXPIParser()
    dexpi = parser.parse_string(xml)

    plant = GraphPlant.from_dexpi(dexpi, backend_class=NetworkXBackend)

    action = ControlAction(
        type=ActionType.SET_VALVE_OPENING,
        target_id="V2",
        value=0.5,
    )

    result = plant.structural_gate([action])
    assert result.status == ValidationStatus.PASS
