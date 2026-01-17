"""Abstract base class for graph backends.

This defines the interface that all graph backend implementations must follow,
enabling the structural gate to work with different graph technologies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple

from anstkit.schemas import ActionType, ControlAction, GateResult, ValidationStatus


class NodeKind(str, Enum):
    """Types of nodes in the plant topology graph."""

    PUMP = "pump"
    VALVE = "valve"
    TANK = "tank"
    SENSOR = "sensor"
    SINK = "sink"
    SOURCE = "source"


class GraphBackend(ABC):
    """Abstract interface for plant topology graphs.

    All graph backends must implement these methods to support
    structural validation in the ANST triad architecture.

    The structural gate uses these methods to:
    1. Verify targets exist in the plant topology
    2. Check that action types match asset kinds
    3. Validate connectivity constraints
    """

    @abstractmethod
    def add_node(self, node_id: str, kind: NodeKind, **attrs) -> None:
        """Add a node to the graph.

        Args:
            node_id: Unique identifier for the node.
            kind: The type of equipment this node represents.
            **attrs: Additional attributes to store on the node.
        """
        pass

    @abstractmethod
    def add_edge(self, source: str, target: str, edge_kind: str, **attrs) -> None:
        """Add a directed edge between nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            edge_kind: Type of relationship (e.g., "flows_to", "controls").
            **attrs: Additional attributes to store on the edge.
        """
        pass

    @abstractmethod
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The node identifier to check.

        Returns:
            True if the node exists, False otherwise.
        """
        pass

    @abstractmethod
    def get_node_kind(self, node_id: str) -> Optional[NodeKind]:
        """Get the kind of a node.

        Args:
            node_id: The node identifier.

        Returns:
            The NodeKind if found, None if node doesn't exist.
        """
        pass

    @abstractmethod
    def has_path(self, source: str, target: str) -> bool:
        """Check if a directed path exists between two nodes.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            True if a path exists from source to target.
        """
        pass

    @abstractmethod
    def get_successors(self, node_id: str) -> List[str]:
        """Get immediate successors of a node.

        Args:
            node_id: The node identifier.

        Returns:
            List of node IDs that this node has edges to.
        """
        pass

    @abstractmethod
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get immediate predecessors of a node.

        Args:
            node_id: The node identifier.

        Returns:
            List of node IDs that have edges to this node.
        """
        pass

    @abstractmethod
    def get_nodes_by_kind(self, kind: NodeKind) -> List[str]:
        """Get all nodes of a specific kind.

        Args:
            kind: The NodeKind to filter by.

        Returns:
            List of node IDs matching the kind.
        """
        pass

    def validate_action(
        self, action: ControlAction, reference_node: str
    ) -> Tuple[bool, List[str]]:
        """Validate a single control action against the graph.

        This is the core structural validation logic that checks:
        1. Target exists in the graph
        2. Action type matches target kind (e.g., SET_PUMP_SPEED on a pump)
        3. Value is within allowed range
        4. Target is connected to the reference node (e.g., the tank)

        Args:
            action: The control action to validate.
            reference_node: A reference node (typically the controlled unit)
                           that the target must be connected to.

        Returns:
            Tuple of (is_valid, list_of_reasons).
        """
        reasons: List[str] = []

        # Check target exists
        if not self.node_exists(action.target_id):
            reasons.append(f"Unknown target_id '{action.target_id}'.")
            return False, reasons

        kind = self.get_node_kind(action.target_id)

        # Check action type matches equipment kind
        if action.type == ActionType.SET_PUMP_SPEED and kind != NodeKind.PUMP:
            reasons.append(
                f"Action {action.type.value} not allowed on kind='{kind.value if kind else 'unknown'}'."
            )

        if action.type == ActionType.SET_VALVE_OPENING and kind != NodeKind.VALVE:
            reasons.append(
                f"Action {action.type.value} not allowed on kind='{kind.value if kind else 'unknown'}'."
            )

        # Range check
        if not (0.0 <= action.value <= 1.0):
            reasons.append(f"Value {action.value:.3f} out of allowed range [0, 1].")

        # Connectivity check: actuator must have path to/from reference
        if kind in {NodeKind.PUMP, NodeKind.VALVE}:
            has_path_to_ref = self.has_path(action.target_id, reference_node)
            has_path_from_ref = self.has_path(reference_node, action.target_id)
            if not (has_path_to_ref or has_path_from_ref):
                reasons.append(
                    f"Actuator '{action.target_id}' not connected to reference node '{reference_node}'."
                )

        return len(reasons) == 0, reasons

    def structural_gate(
        self, actions: List[ControlAction], reference_node: str
    ) -> GateResult:
        """Run structural validation on a list of actions.

        This is the structural gate in the ANST triad. It ensures all
        proposed actions are structurally valid before physics checking.

        Args:
            actions: List of control actions to validate.
            reference_node: The reference node (e.g., tank) for connectivity.

        Returns:
            GateResult with PASS/FAIL status and any failure reasons.
        """
        all_reasons: List[str] = []
        for action in actions:
            ok, reasons = self.validate_action(action, reference_node)
            if not ok:
                all_reasons.extend(reasons)

        return GateResult(
            status=ValidationStatus.PASS if not all_reasons else ValidationStatus.FAIL,
            reasons=all_reasons,
            metrics={"n_actions": len(actions)},
        )
