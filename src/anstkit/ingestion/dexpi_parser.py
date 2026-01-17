"""DEXPI XML parser for P&ID data.

DEXPI (Data Exchange in Process Industry) is an ISO standard for exchanging
P&ID (Process & Instrumentation Diagram) data between engineering tools.

This parser extracts:
- Equipment (pumps, valves, tanks, vessels, etc.)
- Piping connections (flow paths)
- Equipment attributes (descriptions, specifications)

Reference:
- DEXPI specification: https://dexpi.org/
- Based on ISO 15926 data model
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# DEXPI namespace (commonly used versions)
DEXPI_NAMESPACES = {
    "dexpi": "http://www.dexpi.org/schemas/1.3",
    "dexpi12": "http://www.dexpi.org/schemas/1.2",
    "dexpi11": "http://www.dexpi.org/schemas/1.1",
}


@dataclass
class DEXPIEquipment:
    """Represents an equipment item from DEXPI.

    Attributes:
        id: Internal DEXPI ID.
        tag_name: Plant tag name (e.g., P-101, TK-100).
        component_class: Equipment type (Pump, Valve, Tank, etc.).
        description: Human-readable description.
        attributes: Additional attributes from the XML.
    """

    id: str
    tag_name: str
    component_class: str
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DEXPIPipingSegment:
    """Represents a piping connection from DEXPI.

    Attributes:
        id: Internal DEXPI ID.
        from_id: Source equipment ID.
        to_id: Destination equipment ID.
        attributes: Additional attributes (pipe size, spec, etc.).
    """

    id: str
    from_id: str
    to_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DEXPIParseResult:
    """Result of parsing a DEXPI XML file.

    Attributes:
        plant_id: ID of the plant model.
        equipment: List of equipment items.
        piping_segments: List of piping connections.
        warnings: Any parsing warnings.
    """

    plant_id: str
    equipment: List[DEXPIEquipment] = field(default_factory=list)
    piping_segments: List[DEXPIPipingSegment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DEXPIParser:
    """Parser for DEXPI XML format.

    Supports DEXPI versions 1.1, 1.2, and 1.3.

    Example:
        parser = DEXPIParser()
        result = parser.parse_file(Path("plant.xml"))
        for equip in result.equipment:
            print(f"{equip.tag_name}: {equip.component_class}")
    """

    def __init__(self):
        """Initialize the parser."""
        self._namespace: Optional[str] = None

    def parse_file(self, path: Path) -> DEXPIParseResult:
        """Parse a DEXPI XML file.

        Args:
            path: Path to the DEXPI XML file.

        Returns:
            DEXPIParseResult with extracted data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ET.ParseError: If XML is malformed.
        """
        with open(path, "r", encoding="utf-8") as f:
            return self.parse_string(f.read())

    def parse_string(self, xml_content: str) -> DEXPIParseResult:
        """Parse DEXPI XML from a string.

        Args:
            xml_content: XML content as string.

        Returns:
            DEXPIParseResult with extracted data.
        """
        root = ET.fromstring(xml_content)
        self._detect_namespace(root)

        plant_id = root.get("ID", "UnknownPlant")
        result = DEXPIParseResult(plant_id=plant_id)

        # Parse equipment
        result.equipment = self._parse_equipment(root)

        # Parse piping
        result.piping_segments = self._parse_piping(root)

        return result

    def _detect_namespace(self, root: ET.Element) -> None:
        """Detect DEXPI namespace version from root element."""
        tag = root.tag

        # Extract namespace from tag like {http://...}PlantModel
        if tag.startswith("{"):
            ns_end = tag.index("}")
            self._namespace = tag[1:ns_end]
        else:
            # No namespace, use plain tags
            self._namespace = None

    def _ns_tag(self, tag: str) -> str:
        """Create a namespaced tag."""
        if self._namespace:
            return f"{{{self._namespace}}}{tag}"
        return tag

    def _parse_equipment(self, root: ET.Element) -> List[DEXPIEquipment]:
        """Parse equipment elements."""
        equipment = []

        # Find all Equipment elements
        for elem in root.iter():
            # Check both namespaced and non-namespaced
            local_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if local_name == "Equipment":
                equip = self._parse_equipment_element(elem)
                if equip:
                    equipment.append(equip)

        return equipment

    def _parse_equipment_element(self, elem: ET.Element) -> Optional[DEXPIEquipment]:
        """Parse a single Equipment element."""
        equip_id = elem.get("ID")
        if not equip_id:
            return None

        tag_name = elem.get("TagName", equip_id)
        component_class = elem.get("ComponentClass", "Unknown")

        # Extract description from GenericAttributes
        description = ""
        attributes = {}

        for attr_container in elem.iter():
            local_name = (
                attr_container.tag.split("}")[-1]
                if "}" in attr_container.tag
                else attr_container.tag
            )

            if local_name == "GenericAttributes":
                for attr_elem in attr_container:
                    attr_local = (
                        attr_elem.tag.split("}")[-1]
                        if "}" in attr_elem.tag
                        else attr_elem.tag
                    )
                    if attr_local == "Attribute":
                        attr_name = attr_elem.get("Name", "")
                        attr_value = attr_elem.get("Value", "")
                        if attr_name.lower() == "description":
                            description = attr_value
                        else:
                            attributes[attr_name] = attr_value

        return DEXPIEquipment(
            id=equip_id,
            tag_name=tag_name,
            component_class=component_class,
            description=description,
            attributes=attributes,
        )

    def _parse_piping(self, root: ET.Element) -> List[DEXPIPipingSegment]:
        """Parse piping network segments."""
        segments = []

        for elem in root.iter():
            local_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if local_name == "PipingNetworkSegment":
                segment = self._parse_piping_segment(elem)
                if segment:
                    segments.append(segment)

        return segments

    def _parse_piping_segment(self, elem: ET.Element) -> Optional[DEXPIPipingSegment]:
        """Parse a single PipingNetworkSegment element."""
        segment_id = elem.get("ID")
        from_id = elem.get("FromID")
        to_id = elem.get("ToID")

        if not (segment_id and from_id and to_id):
            return None

        return DEXPIPipingSegment(
            id=segment_id,
            from_id=from_id,
            to_id=to_id,
            attributes={},
        )
