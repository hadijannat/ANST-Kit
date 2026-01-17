"""GraphRAG-enhanced graph backend.

This module provides a graph backend that can integrate with Microsoft GraphRAG
for semantic querying, community detection, and evidence retrieval.

When graphrag is not available or no index is provided, it falls back to
a NetworkX-based implementation with simpler heuristics.

Usage:
    # Without GraphRAG (fallback mode)
    backend = GraphRAGBackend(index_dir=None)
    backend.add_node("P1", NodeKind.PUMP)

    # With GraphRAG index
    backend = GraphRAGBackend(index_dir=Path("./graphrag_output"))
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from anstkit.schemas import ControlAction, GateResult

from .base import GraphBackend, NodeKind
from .networkx_backend import NetworkXBackend

logger = logging.getLogger(__name__)

# Check if graphrag is available
GRAPHRAG_AVAILABLE = importlib.util.find_spec("graphrag") is not None

if GRAPHRAG_AVAILABLE:
    import pandas as pd
else:
    pd = None


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG integration.

    Attributes:
        index_dir: Directory containing GraphRAG index files.
        community_level: Hierarchical community level for queries.
        enable_semantic_search: Whether to use LLM-based semantic search.
    """

    index_dir: Optional[Path] = None
    community_level: int = 2
    enable_semantic_search: bool = True


class GraphRAGBackend(GraphBackend):
    """Graph backend with optional GraphRAG RAG capabilities.

    This backend maintains a dual representation:
    1. NetworkX graph for fast structural queries
    2. GraphRAG index for semantic queries (when available)

    The structural gate always uses the NetworkX representation for
    deterministic, fast validation. RAG features are used for:
    - Semantic search ("What equipment handles cooling?")
    - Community detection (grouping related equipment)
    - Evidence retrieval (explaining structural decisions)
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        config: Optional[GraphRAGConfig] = None,
    ):
        """Initialize GraphRAG backend.

        Args:
            index_dir: Path to GraphRAG index directory. If None, runs in
                      fallback mode using only NetworkX.
            config: Additional configuration options.
        """
        self._config = config or GraphRAGConfig(index_dir=index_dir)
        self._fallback = NetworkXBackend()
        self._graphrag_loaded = False
        self._entities_df = None
        self._relationships_df = None
        self._communities_df = None
        self._community_reports_df = None

        if index_dir and GRAPHRAG_AVAILABLE:
            self._load_graphrag_index(index_dir)

    def _load_graphrag_index(self, index_dir: Path) -> None:
        """Load GraphRAG index files if available."""
        try:
            entities_path = index_dir / "entities.parquet"
            relationships_path = index_dir / "relationships.parquet"
            communities_path = index_dir / "communities.parquet"
            community_reports_path = index_dir / "community_reports.parquet"

            if entities_path.exists():
                self._entities_df = pd.read_parquet(entities_path)
                logger.info(f"Loaded {len(self._entities_df)} entities from GraphRAG")

            if relationships_path.exists():
                self._relationships_df = pd.read_parquet(relationships_path)
                logger.info(
                    f"Loaded {len(self._relationships_df)} relationships from GraphRAG"
                )

            if communities_path.exists():
                self._communities_df = pd.read_parquet(communities_path)

            if community_reports_path.exists():
                self._community_reports_df = pd.read_parquet(community_reports_path)

            self._graphrag_loaded = True
            logger.info(f"GraphRAG index loaded from {index_dir}")

        except Exception as e:
            logger.warning(f"Failed to load GraphRAG index: {e}. Using fallback mode.")
            self._graphrag_loaded = False

    # =========================================================================
    # GraphBackend interface implementation (delegates to NetworkX fallback)
    # =========================================================================

    def add_node(self, node_id: str, kind: NodeKind, **attrs) -> None:
        """Add a node to the graph."""
        self._fallback.add_node(node_id, kind, **attrs)

    def add_edge(self, source: str, target: str, edge_kind: str, **attrs) -> None:
        """Add a directed edge between nodes."""
        self._fallback.add_edge(source, target, edge_kind, **attrs)

    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists."""
        return self._fallback.node_exists(node_id)

    def get_node_kind(self, node_id: str) -> Optional[NodeKind]:
        """Get the kind of a node."""
        return self._fallback.get_node_kind(node_id)

    def has_path(self, source: str, target: str) -> bool:
        """Check if a directed path exists between nodes."""
        return self._fallback.has_path(source, target)

    def get_successors(self, node_id: str) -> List[str]:
        """Get immediate successors of a node."""
        return self._fallback.get_successors(node_id)

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get immediate predecessors of a node."""
        return self._fallback.get_predecessors(node_id)

    def get_nodes_by_kind(self, kind: NodeKind) -> List[str]:
        """Get all nodes of a specific kind."""
        return self._fallback.get_nodes_by_kind(kind)

    # Override structural_gate to use the fallback's validation
    def structural_gate(
        self, actions: List[ControlAction], reference_node: str
    ) -> GateResult:
        """Run structural validation using the fallback backend."""
        return super().structural_gate(actions, reference_node)

    # =========================================================================
    # RAG-enhanced features
    # =========================================================================

    def semantic_query(self, query: str) -> Dict[str, Any]:
        """Perform a semantic query over the graph.

        Uses GraphRAG's global/local search when available, otherwise
        falls back to keyword matching on node attributes.

        Args:
            query: Natural language query about the plant topology.

        Returns:
            Dict with 'answer', 'sources', and 'confidence' keys.
        """
        if self._graphrag_loaded and self._config.enable_semantic_search:
            return self._graphrag_semantic_search(query)
        else:
            return self._fallback_semantic_search(query)

    def _graphrag_semantic_search(self, query: str) -> Dict[str, Any]:
        """Perform semantic search using GraphRAG (async wrapper)."""
        # Note: Full GraphRAG search requires async and LLM config
        # For now, return a structured response based on indexed data
        if self._entities_df is not None and not self._entities_df.empty:
            # Simple keyword matching on entity descriptions
            query_lower = query.lower()
            matches = []
            for _, row in self._entities_df.iterrows():
                desc = str(row.get("description", "")).lower()
                name = str(row.get("name", "")).lower()
                if query_lower in desc or query_lower in name:
                    matches.append(row.get("name", ""))

            return {
                "answer": f"Found {len(matches)} relevant entities: {', '.join(matches[:5])}",
                "sources": matches[:10],
                "confidence": 0.7 if matches else 0.3,
                "mode": "graphrag_keyword",
            }

        return self._fallback_semantic_search(query)

    def _fallback_semantic_search(self, query: str) -> Dict[str, Any]:
        """Simple keyword-based search over node attributes."""
        query_lower = query.lower()
        matches = []

        for node_id in self._fallback.get_all_nodes():
            attrs = self._fallback.get_node_attrs(node_id)
            desc = str(attrs.get("description", "")).lower()
            kind = str(attrs.get("kind", "")).lower()

            if query_lower in node_id.lower() or query_lower in desc or query_lower in kind:
                matches.append(node_id)

        if matches:
            return {
                "answer": f"Found nodes matching query: {', '.join(matches)}",
                "sources": matches,
                "confidence": 0.5,
                "mode": "fallback_keyword",
            }

        return {
            "answer": "No matching nodes found. Try a more specific query.",
            "sources": [],
            "confidence": 0.1,
            "mode": "fallback_keyword",
        }

    def get_connected_equipment(self, reference_node: str) -> List[str]:
        """Get all equipment connected to a reference node.

        Traverses the graph in both directions to find all equipment
        that has a flow path to or from the reference node.

        Args:
            reference_node: The node ID to find connections for.

        Returns:
            List of connected node IDs.
        """
        connected: Set[str] = set()

        # BFS in both directions
        for node_id in self._fallback.get_all_nodes():
            if node_id == reference_node:
                continue
            if self.has_path(node_id, reference_node) or self.has_path(
                reference_node, node_id
            ):
                connected.add(node_id)

        return list(connected)

    def get_communities(self) -> List[List[str]]:
        """Get community groupings of nodes.

        Uses GraphRAG community detection when available, otherwise
        returns a single community containing all nodes.

        Returns:
            List of communities, where each community is a list of node IDs.
        """
        if self._graphrag_loaded and self._communities_df is not None:
            return self._extract_graphrag_communities()

        # Fallback: single community with all nodes
        return [self._fallback.get_all_nodes()]

    def _extract_graphrag_communities(self) -> List[List[str]]:
        """Extract communities from GraphRAG community data."""
        if self._communities_df is None or self._communities_df.empty:
            return [self._fallback.get_all_nodes()]

        communities = []
        # Group by community ID at the configured level
        try:
            for _, group in self._communities_df.groupby("community"):
                members = group.get("title", group.get("id", [])).tolist()
                if members:
                    communities.append(members)
        except Exception as e:
            logger.warning(f"Failed to extract communities: {e}")
            communities = [self._fallback.get_all_nodes()]

        return communities if communities else [self._fallback.get_all_nodes()]

    def get_entity_summary(self, node_id: str) -> str:
        """Get a summary description of an entity.

        Uses GraphRAG entity descriptions when available, otherwise
        constructs a summary from node attributes.

        Args:
            node_id: The node to summarize.

        Returns:
            Human-readable summary string.
        """
        if self._graphrag_loaded and self._entities_df is not None:
            # Look up entity in GraphRAG index
            matches = self._entities_df[
                self._entities_df["name"].str.lower() == node_id.lower()
            ]
            if not matches.empty:
                desc = matches.iloc[0].get("description", "")
                if desc:
                    return str(desc)

        # Fallback: construct from node attributes
        if not self.node_exists(node_id):
            return f"Unknown entity: {node_id}"

        kind = self.get_node_kind(node_id)
        attrs = self._fallback.get_node_attrs(node_id)
        desc = attrs.get("description", "")

        summary = f"{node_id} is a {kind.value if kind else 'component'}"
        if desc:
            summary += f": {desc}"

        # Add connectivity info
        predecessors = self.get_predecessors(node_id)
        successors = self.get_successors(node_id)
        if predecessors:
            summary += f". Receives from: {', '.join(predecessors)}"
        if successors:
            summary += f". Feeds: {', '.join(successors)}"

        return summary

    def get_structural_evidence(
        self, source: str, target: str
    ) -> Dict[str, Any]:
        """Get evidence explaining the structural relationship between nodes.

        Useful for explainability in the structural gate - explains WHY
        a validation passed or failed.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            Dict with 'path', 'relationship', and 'explanation' keys.
        """
        evidence: Dict[str, Any] = {
            "path": [],
            "relationship": "unknown",
            "explanation": "",
        }

        if not self.node_exists(source):
            evidence["explanation"] = f"Source node '{source}' does not exist"
            return evidence

        if not self.node_exists(target):
            evidence["explanation"] = f"Target node '{target}' does not exist"
            return evidence

        # Check direct connection
        if target in self.get_successors(source):
            evidence["path"] = [source, target]
            evidence["relationship"] = "direct"
            evidence["explanation"] = f"{source} directly connects to {target}"
            return evidence

        # Check path existence
        if self.has_path(source, target):
            # Find path using NetworkX
            import networkx as nx

            nx_graph = self._fallback.to_networkx()
            try:
                path = nx.shortest_path(nx_graph, source, target)
                evidence["path"] = path
                evidence["relationship"] = "indirect"
                evidence["explanation"] = (
                    f"{source} connects to {target} via path: {' -> '.join(path)}"
                )
            except nx.NetworkXNoPath:
                evidence["relationship"] = "none"
                evidence["explanation"] = f"No path from {source} to {target}"
        else:
            evidence["relationship"] = "none"
            evidence["explanation"] = f"No path exists from {source} to {target}"

        return evidence

    @property
    def is_graphrag_enabled(self) -> bool:
        """Check if GraphRAG features are available."""
        return self._graphrag_loaded

    @property
    def mode(self) -> str:
        """Get current operating mode."""
        if self._graphrag_loaded:
            return "graphrag"
        return "fallback"
