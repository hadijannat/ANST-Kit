"""NetworkX implementation of the graph backend.

This is the default backend for development and demos. It provides
a full-featured graph implementation using NetworkX with no external
dependencies beyond the Python ecosystem.

For production deployments requiring:
- Persistence: Consider Neo4j or Memgraph backends
- RAG capabilities: Use the GraphRAG backend
- Distributed scale: Use a graph database backend
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from .base import GraphBackend, NodeKind


class NetworkXBackend(GraphBackend):
    """NetworkX-based implementation of the graph backend.

    Uses an in-memory directed graph suitable for development,
    testing, and small-scale deployments.
    """

    def __init__(self):
        """Initialize an empty graph."""
        self._graph = nx.DiGraph()

    def add_node(self, node_id: str, kind: NodeKind, **attrs) -> None:
        """Add a node with its kind and optional attributes."""
        self._graph.add_node(node_id, kind=kind, **attrs)

    def add_edge(self, source: str, target: str, edge_kind: str, **attrs) -> None:
        """Add a directed edge between nodes."""
        self._graph.add_edge(source, target, kind=edge_kind, **attrs)

    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._graph

    def get_node_kind(self, node_id: str) -> Optional[NodeKind]:
        """Get the kind of a node, or None if it doesn't exist."""
        if node_id not in self._graph:
            return None
        kind_value = self._graph.nodes[node_id].get("kind")
        if isinstance(kind_value, NodeKind):
            return kind_value
        if isinstance(kind_value, str):
            try:
                return NodeKind(kind_value)
            except ValueError:
                return None
        return None

    def has_path(self, source: str, target: str) -> bool:
        """Check if a directed path exists between nodes."""
        if source not in self._graph or target not in self._graph:
            return False
        return nx.has_path(self._graph, source, target)

    def get_successors(self, node_id: str) -> List[str]:
        """Get immediate successors of a node."""
        if node_id not in self._graph:
            return []
        return list(self._graph.successors(node_id))

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get immediate predecessors of a node."""
        if node_id not in self._graph:
            return []
        return list(self._graph.predecessors(node_id))

    def get_nodes_by_kind(self, kind: NodeKind) -> List[str]:
        """Get all nodes of a specific kind."""
        return [
            node_id
            for node_id, attrs in self._graph.nodes(data=True)
            if attrs.get("kind") == kind
        ]

    def get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        """Get all attributes of a node.

        Args:
            node_id: The node identifier.

        Returns:
            Dictionary of node attributes, empty if node doesn't exist.
        """
        if node_id not in self._graph:
            return {}
        return dict(self._graph.nodes[node_id])

    def get_edge_attrs(self, source: str, target: str) -> Dict[str, Any]:
        """Get all attributes of an edge.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            Dictionary of edge attributes, empty if edge doesn't exist.
        """
        if not self._graph.has_edge(source, target):
            return {}
        return dict(self._graph.edges[source, target])

    def get_all_nodes(self) -> List[str]:
        """Get all node IDs in the graph."""
        return list(self._graph.nodes())

    def get_all_edges(self) -> List[tuple]:
        """Get all edges as (source, target) tuples."""
        return list(self._graph.edges())

    def subgraph(self, node_ids: List[str]) -> "NetworkXBackend":
        """Create a new backend containing only the specified nodes.

        Args:
            node_ids: List of node IDs to include.

        Returns:
            New NetworkXBackend with the subgraph.
        """
        new_backend = NetworkXBackend()
        new_backend._graph = self._graph.subgraph(node_ids).copy()
        return new_backend

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph) -> "NetworkXBackend":
        """Create a backend from an existing NetworkX graph.

        The graph nodes must have a 'kind' attribute that maps to NodeKind.

        Args:
            graph: A NetworkX DiGraph.

        Returns:
            NetworkXBackend wrapping the graph.
        """
        backend = cls()
        backend._graph = graph.copy()
        return backend

    def to_networkx(self) -> nx.DiGraph:
        """Export the internal graph as a NetworkX DiGraph.

        Returns:
            A copy of the internal graph.
        """
        return self._graph.copy()
