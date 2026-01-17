"""Tests for GraphRAG backend integration.

These tests verify the GraphRAG backend interface. Full integration tests
require graphrag to be installed and configured.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from anstkit.graph.base import NodeKind
from anstkit.graph.networkx_backend import NetworkXBackend
from anstkit.schemas import ActionType, ControlAction, ValidationStatus


class TestGraphRAGBackendCore:
    """Test GraphRAG backend basic functionality."""

    def test_graphrag_backend_import(self):
        """GraphRAGBackend can be imported (even without graphrag installed)."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

    def test_graphrag_backend_with_networkx_fallback(self):
        """GraphRAGBackend works with NetworkX fallback when no index."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

        backend = GraphRAGBackend(index_dir=None)  # No index = fallback mode
        backend.add_node("P1", NodeKind.PUMP)
        backend.add_node("T1", NodeKind.TANK)
        backend.add_edge("P1", "T1", "flows_to")

        assert backend.node_exists("P1")
        assert backend.get_node_kind("P1") == NodeKind.PUMP
        assert backend.has_path("P1", "T1")

    def test_structural_gate_works_in_fallback(self):
        """Structural gate works with fallback backend."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

        backend = GraphRAGBackend(index_dir=None)
        backend.add_node("P1", NodeKind.PUMP)
        backend.add_node("V1", NodeKind.VALVE)
        backend.add_node("T1", NodeKind.TANK)
        backend.add_edge("P1", "V1", "flows_to")
        backend.add_edge("V1", "T1", "flows_to")

        actions = [
            ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.5),
        ]
        result = backend.structural_gate(actions, "T1")
        assert result.status == ValidationStatus.PASS

    def test_semantic_query_interface(self):
        """Semantic query interface exists (returns fallback in mock mode)."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

        backend = GraphRAGBackend(index_dir=None)
        backend.add_node("P1", NodeKind.PUMP, description="Main inlet pump")
        backend.add_node("T1", NodeKind.TANK, description="Primary storage tank")
        backend.add_edge("P1", "T1", "flows_to")

        # Semantic query should work (returns basic results without LLM)
        results = backend.semantic_query("What pumps feed tank T1?")
        assert isinstance(results, dict)
        assert "answer" in results

    def test_get_connected_equipment(self):
        """Can retrieve equipment connected to a reference node."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

        backend = GraphRAGBackend(index_dir=None)
        backend.add_node("P1", NodeKind.PUMP)
        backend.add_node("V1", NodeKind.VALVE)
        backend.add_node("T1", NodeKind.TANK)
        backend.add_node("V2", NodeKind.VALVE)
        backend.add_edge("P1", "V1", "flows_to")
        backend.add_edge("V1", "T1", "flows_to")
        backend.add_edge("T1", "V2", "flows_to")

        connected = backend.get_connected_equipment("T1")
        # Should return all nodes reachable to/from T1
        assert "P1" in connected
        assert "V1" in connected
        assert "V2" in connected


class TestGraphRAGBackendRAGFeatures:
    """Test RAG-specific features (require graphrag or use mocks)."""

    def test_community_detection_interface(self):
        """Community detection interface exists."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

        backend = GraphRAGBackend(index_dir=None)
        backend.add_node("P1", NodeKind.PUMP)
        backend.add_node("T1", NodeKind.TANK)
        backend.add_edge("P1", "T1", "flows_to")

        communities = backend.get_communities()
        assert isinstance(communities, list)
        # In fallback mode, returns single community with all nodes
        assert len(communities) >= 1

    def test_get_entity_summary_interface(self):
        """Entity summary interface exists."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

        backend = GraphRAGBackend(index_dir=None)
        backend.add_node("P1", NodeKind.PUMP, description="Main pump unit")

        summary = backend.get_entity_summary("P1")
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_evidence_retrieval_interface(self):
        """Evidence retrieval interface exists for explainability."""
        from anstkit.graph.graphrag_backend import GraphRAGBackend

        backend = GraphRAGBackend(index_dir=None)
        backend.add_node("P1", NodeKind.PUMP)
        backend.add_node("T1", NodeKind.TANK)
        backend.add_edge("P1", "T1", "flows_to")

        evidence = backend.get_structural_evidence("P1", "T1")
        assert isinstance(evidence, dict)
        assert "path" in evidence
        assert "relationship" in evidence
