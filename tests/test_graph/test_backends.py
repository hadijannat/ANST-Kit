"""Tests for graph backend implementations."""

import pytest
from abc import ABC

from anstkit.graph.base import GraphBackend, NodeKind
from anstkit.graph.networkx_backend import NetworkXBackend
from anstkit.schemas import ActionType, ControlAction, ValidationStatus


class TestGraphBackendInterface:
    """Test that the abstract interface is properly defined."""

    def test_graph_backend_is_abstract(self):
        """GraphBackend cannot be instantiated directly."""
        assert issubclass(GraphBackend, ABC)
        with pytest.raises(TypeError):
            GraphBackend()


class TestNetworkXBackend:
    """Test NetworkX implementation of graph backend."""

    @pytest.fixture
    def backend(self) -> NetworkXBackend:
        """Create a backend with demo plant topology."""
        backend = NetworkXBackend()
        backend.add_node("P1", NodeKind.PUMP)
        backend.add_node("V1", NodeKind.VALVE)
        backend.add_node("T1", NodeKind.TANK)
        backend.add_node("V2", NodeKind.VALVE)
        backend.add_node("SINK", NodeKind.SINK)
        backend.add_edge("P1", "V1", "flows_to")
        backend.add_edge("V1", "T1", "flows_to")
        backend.add_edge("T1", "V2", "flows_to")
        backend.add_edge("V2", "SINK", "flows_to")
        return backend

    def test_node_exists(self, backend: NetworkXBackend):
        """Can query whether nodes exist."""
        assert backend.node_exists("P1")
        assert backend.node_exists("T1")
        assert not backend.node_exists("NONEXISTENT")

    def test_get_node_kind(self, backend: NetworkXBackend):
        """Can retrieve node kind."""
        assert backend.get_node_kind("P1") == NodeKind.PUMP
        assert backend.get_node_kind("V1") == NodeKind.VALVE
        assert backend.get_node_kind("T1") == NodeKind.TANK

    def test_has_path(self, backend: NetworkXBackend):
        """Can check path connectivity."""
        assert backend.has_path("P1", "T1")
        assert backend.has_path("T1", "SINK")
        assert not backend.has_path("SINK", "P1")  # Wrong direction

    def test_validate_action_unknown_target(self, backend: NetworkXBackend):
        """Rejects action targeting unknown node."""
        action = ControlAction(
            type=ActionType.SET_PUMP_SPEED,
            target_id="NONEXISTENT",
            value=0.5,
        )
        ok, reasons = backend.validate_action(action, "T1")
        assert not ok
        assert any("Unknown" in r or "unknown" in r.lower() for r in reasons)

    def test_validate_action_wrong_type(self, backend: NetworkXBackend):
        """Rejects pump action on valve."""
        action = ControlAction(
            type=ActionType.SET_PUMP_SPEED,
            target_id="V1",
            value=0.5,
        )
        ok, reasons = backend.validate_action(action, "T1")
        assert not ok
        assert any("not allowed" in r.lower() for r in reasons)

    def test_validate_action_out_of_range(self, backend: NetworkXBackend):
        """Rejects out-of-range values."""
        action = ControlAction(
            type=ActionType.SET_PUMP_SPEED,
            target_id="P1",
            value=1.5,  # > 1.0
        )
        ok, reasons = backend.validate_action(action, "T1")
        assert not ok
        assert any("range" in r.lower() for r in reasons)

    def test_validate_action_valid(self, backend: NetworkXBackend):
        """Accepts valid action."""
        action = ControlAction(
            type=ActionType.SET_PUMP_SPEED,
            target_id="P1",
            value=0.7,
        )
        ok, reasons = backend.validate_action(action, "T1")
        assert ok
        assert len(reasons) == 0

    def test_structural_gate(self, backend: NetworkXBackend):
        """Structural gate aggregates validation results."""
        actions = [
            ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7),
            ControlAction(type=ActionType.SET_VALVE_OPENING, target_id="V1", value=0.5),
        ]
        result = backend.structural_gate(actions, "T1")
        assert result.status == ValidationStatus.PASS

    def test_structural_gate_with_invalid(self, backend: NetworkXBackend):
        """Structural gate fails on any invalid action."""
        actions = [
            ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7),
            ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="INVALID", value=0.5),
        ]
        result = backend.structural_gate(actions, "T1")
        assert result.status == ValidationStatus.FAIL

    def test_get_neighbors(self, backend: NetworkXBackend):
        """Can retrieve node neighbors."""
        successors = backend.get_successors("V1")
        predecessors = backend.get_predecessors("V1")
        assert "T1" in successors
        assert "P1" in predecessors

    def test_get_all_nodes_of_kind(self, backend: NetworkXBackend):
        """Can retrieve all nodes of a given kind."""
        valves = backend.get_nodes_by_kind(NodeKind.VALVE)
        assert set(valves) == {"V1", "V2"}
