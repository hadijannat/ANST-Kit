"""Topology extractor for converting DEXPI data to graph backends.

This module bridges the DEXPI parser output with the graph backend
abstraction, enabling seamless loading of industrial P&ID data into
the ANST-Kit structural validation system.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

from anstkit.graph.base import GraphBackend, NodeKind
from anstkit.ingestion.dexpi_parser import DEXPIParseResult

logger = logging.getLogger(__name__)


# Mapping from DEXPI ComponentClass to NodeKind
COMPONENT_CLASS_MAP: Dict[str, NodeKind] = {
    # Pumps
    "Pump": NodeKind.PUMP,
    "CentrifugalPump": NodeKind.PUMP,
    "PositiveDisplacementPump": NodeKind.PUMP,
    "VacuumPump": NodeKind.PUMP,
    # Valves
    "Valve": NodeKind.VALVE,
    "ControlValve": NodeKind.VALVE,
    "GateValve": NodeKind.VALVE,
    "BallValve": NodeKind.VALVE,
    "CheckValve": NodeKind.VALVE,
    "GlobeValve": NodeKind.VALVE,
    "ButterflyValve": NodeKind.VALVE,
    "ReliefValve": NodeKind.VALVE,
    # Tanks and vessels
    "Tank": NodeKind.TANK,
    "StorageTank": NodeKind.TANK,
    "Vessel": NodeKind.TANK,
    "PressureVessel": NodeKind.TANK,
    "Reactor": NodeKind.TANK,
    "Column": NodeKind.TANK,
    "Drum": NodeKind.TANK,
    # Sensors and instruments
    "Sensor": NodeKind.SENSOR,
    "Instrument": NodeKind.SENSOR,
    "Transmitter": NodeKind.SENSOR,
    "FlowMeter": NodeKind.SENSOR,
    "LevelIndicator": NodeKind.SENSOR,
    "PressureIndicator": NodeKind.SENSOR,
    "TemperatureIndicator": NodeKind.SENSOR,
    # Sources and sinks
    "Source": NodeKind.SOURCE,
    "Sink": NodeKind.SINK,
    "Drain": NodeKind.SINK,
    "Vent": NodeKind.SINK,
}


class TopologyExtractor:
    """Extract plant topology from DEXPI data into graph backends.

    This class handles the conversion from DEXPI's P&ID representation
    to ANST-Kit's graph-based structural model.

    Example:
        parser = DEXPIParser()
        dexpi_data = parser.parse_file(Path("plant.xml"))

        extractor = TopologyExtractor()
        backend = extractor.to_graph_backend(dexpi_data, NetworkXBackend)
    """

    def __init__(self, class_map: Dict[str, NodeKind] | None = None):
        """Initialize extractor.

        Args:
            class_map: Custom mapping from DEXPI ComponentClass to NodeKind.
                      Uses default COMPONENT_CLASS_MAP if not provided.
        """
        self._class_map = class_map or COMPONENT_CLASS_MAP

    def to_graph_backend(
        self,
        dexpi_data: DEXPIParseResult,
        backend_class: Type[GraphBackend],
    ) -> GraphBackend:
        """Convert DEXPI parse result to a graph backend instance.

        Args:
            dexpi_data: Parsed DEXPI data.
            backend_class: The GraphBackend subclass to instantiate.

        Returns:
            Populated GraphBackend instance.
        """
        backend = backend_class()

        # Add equipment as nodes
        for equip in dexpi_data.equipment:
            node_kind = self._map_component_class(equip.component_class)
            backend.add_node(
                equip.tag_name,
                node_kind,
                description=equip.description,
                dexpi_id=equip.id,
                component_class=equip.component_class,
                **equip.attributes,
            )
            logger.debug(
                f"Added node {equip.tag_name} as {node_kind.value} "
                f"(DEXPI class: {equip.component_class})"
            )

        # Add piping as edges
        for segment in dexpi_data.piping_segments:
            # Validate that both endpoints exist
            if not backend.node_exists(segment.from_id):
                logger.warning(
                    f"Piping segment {segment.id} references unknown source: {segment.from_id}"
                )
                continue
            if not backend.node_exists(segment.to_id):
                logger.warning(
                    f"Piping segment {segment.id} references unknown target: {segment.to_id}"
                )
                continue

            backend.add_edge(
                segment.from_id,
                segment.to_id,
                "flows_to",
                dexpi_id=segment.id,
                **segment.attributes,
            )
            logger.debug(
                f"Added edge {segment.from_id} -> {segment.to_id} (segment {segment.id})"
            )

        return backend

    def _map_component_class(self, component_class: str) -> NodeKind:
        """Map DEXPI ComponentClass to NodeKind.

        Args:
            component_class: DEXPI component class string.

        Returns:
            Corresponding NodeKind (defaults to SENSOR for unknown types).
        """
        # Try exact match first
        if component_class in self._class_map:
            return self._class_map[component_class]

        # Try case-insensitive match
        class_lower = component_class.lower()
        for key, value in self._class_map.items():
            if key.lower() == class_lower:
                return value

        # Try partial match (e.g., "CentrifugalPump" contains "Pump")
        for key, value in self._class_map.items():
            if key.lower() in class_lower or class_lower in key.lower():
                return value

        logger.warning(
            f"Unknown component class '{component_class}', defaulting to SENSOR"
        )
        return NodeKind.SENSOR

    def get_tank_reference_node(self, backend: GraphBackend) -> str | None:
        """Find a suitable tank to use as reference node.

        For structural validation, we typically need a reference node
        (usually the controlled tank/vessel). This method finds the
        first tank in the graph.

        Args:
            backend: The populated graph backend.

        Returns:
            Tag name of a tank, or None if no tanks found.
        """
        tanks = backend.get_nodes_by_kind(NodeKind.TANK)
        return tanks[0] if tanks else None
