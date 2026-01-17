"""Tests for DEXPI XML parsing."""


from anstkit.graph.base import NodeKind
from anstkit.ingestion.dexpi_parser import DEXPIParser

# Sample DEXPI-like XML for testing
SAMPLE_DEXPI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PlantModel xmlns="http://www.dexpi.org/schemas/1.3"
            ID="TestPlant">
  <Equipment ID="P-101" TagName="P-101" ComponentClass="Pump">
    <GenericAttributes>
      <Attribute Name="Description" Value="Main Feed Pump"/>
    </GenericAttributes>
  </Equipment>
  <Equipment ID="V-101" TagName="V-101" ComponentClass="Valve">
    <GenericAttributes>
      <Attribute Name="Description" Value="Inlet Control Valve"/>
    </GenericAttributes>
  </Equipment>
  <Equipment ID="TK-101" TagName="TK-101" ComponentClass="Tank">
    <GenericAttributes>
      <Attribute Name="Description" Value="Primary Storage Tank"/>
    </GenericAttributes>
  </Equipment>
  <Equipment ID="V-102" TagName="V-102" ComponentClass="Valve">
    <GenericAttributes>
      <Attribute Name="Description" Value="Outlet Control Valve"/>
    </GenericAttributes>
  </Equipment>
  <PipingNetworkSystem ID="PipingSystem1">
    <PipingNetworkSegment ID="PipeSegment1" FromID="P-101" ToID="V-101"/>
    <PipingNetworkSegment ID="PipeSegment2" FromID="V-101" ToID="TK-101"/>
    <PipingNetworkSegment ID="PipeSegment3" FromID="TK-101" ToID="V-102"/>
  </PipingNetworkSystem>
</PlantModel>
"""


class TestDEXPIParser:
    """Test DEXPI XML parsing."""

    def test_parser_can_be_instantiated(self):
        """Parser can be created."""
        parser = DEXPIParser()
        assert parser is not None

    def test_parse_equipment_from_xml(self):
        """Parser extracts equipment from XML."""
        parser = DEXPIParser()
        result = parser.parse_string(SAMPLE_DEXPI_XML)

        assert len(result.equipment) == 4
        tags = {e.tag_name for e in result.equipment}
        assert "P-101" in tags
        assert "V-101" in tags
        assert "TK-101" in tags
        assert "V-102" in tags

    def test_parse_equipment_classes(self):
        """Parser correctly identifies equipment classes."""
        parser = DEXPIParser()
        result = parser.parse_string(SAMPLE_DEXPI_XML)

        equipment_by_tag = {e.tag_name: e for e in result.equipment}

        assert equipment_by_tag["P-101"].component_class == "Pump"
        assert equipment_by_tag["V-101"].component_class == "Valve"
        assert equipment_by_tag["TK-101"].component_class == "Tank"

    def test_parse_piping_connections(self):
        """Parser extracts piping connections."""
        parser = DEXPIParser()
        result = parser.parse_string(SAMPLE_DEXPI_XML)

        assert len(result.piping_segments) == 3

        # Check connectivity
        connections = [(s.from_id, s.to_id) for s in result.piping_segments]
        assert ("P-101", "V-101") in connections
        assert ("V-101", "TK-101") in connections
        assert ("TK-101", "V-102") in connections

    def test_parse_descriptions(self):
        """Parser extracts equipment descriptions."""
        parser = DEXPIParser()
        result = parser.parse_string(SAMPLE_DEXPI_XML)

        equipment_by_tag = {e.tag_name: e for e in result.equipment}
        assert "Main Feed Pump" in equipment_by_tag["P-101"].description
        assert "Storage Tank" in equipment_by_tag["TK-101"].description


class TestTopologyExtractor:
    """Test conversion from DEXPI to graph backend."""

    def test_extract_to_graph_backend(self):
        """Can convert DEXPI parse result to graph backend."""
        from anstkit.graph.networkx_backend import NetworkXBackend
        from anstkit.ingestion.topology_extractor import TopologyExtractor

        parser = DEXPIParser()
        dexpi_result = parser.parse_string(SAMPLE_DEXPI_XML)

        extractor = TopologyExtractor()
        backend = extractor.to_graph_backend(dexpi_result, NetworkXBackend)

        # Verify nodes were created
        assert backend.node_exists("P-101")
        assert backend.node_exists("V-101")
        assert backend.node_exists("TK-101")
        assert backend.node_exists("V-102")

    def test_node_kinds_mapped_correctly(self):
        """Equipment classes map to NodeKind correctly."""
        from anstkit.graph.networkx_backend import NetworkXBackend
        from anstkit.ingestion.topology_extractor import TopologyExtractor

        parser = DEXPIParser()
        dexpi_result = parser.parse_string(SAMPLE_DEXPI_XML)

        extractor = TopologyExtractor()
        backend = extractor.to_graph_backend(dexpi_result, NetworkXBackend)

        assert backend.get_node_kind("P-101") == NodeKind.PUMP
        assert backend.get_node_kind("V-101") == NodeKind.VALVE
        assert backend.get_node_kind("TK-101") == NodeKind.TANK

    def test_edges_created_from_piping(self):
        """Piping segments create graph edges."""
        from anstkit.graph.networkx_backend import NetworkXBackend
        from anstkit.ingestion.topology_extractor import TopologyExtractor

        parser = DEXPIParser()
        dexpi_result = parser.parse_string(SAMPLE_DEXPI_XML)

        extractor = TopologyExtractor()
        backend = extractor.to_graph_backend(dexpi_result, NetworkXBackend)

        # Verify connectivity
        assert backend.has_path("P-101", "TK-101")
        assert backend.has_path("TK-101", "V-102")
        assert "V-101" in backend.get_successors("P-101")

    def test_descriptions_preserved(self):
        """Equipment descriptions are stored as node attributes."""
        from anstkit.graph.networkx_backend import NetworkXBackend
        from anstkit.ingestion.topology_extractor import TopologyExtractor

        parser = DEXPIParser()
        dexpi_result = parser.parse_string(SAMPLE_DEXPI_XML)

        extractor = TopologyExtractor()
        backend = extractor.to_graph_backend(dexpi_result, NetworkXBackend)

        attrs = backend.get_node_attrs("P-101")
        assert "description" in attrs
        assert "Main Feed Pump" in attrs["description"]
