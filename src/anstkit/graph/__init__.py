"""Graph backend abstraction for ANST-Kit.

This module provides a pluggable graph backend interface that enables
switching between different graph implementations:
- NetworkX (demo/development)
- Microsoft GraphRAG (production with RAG capabilities)
- Neo4j/Memgraph (enterprise graph databases)
"""

from .base import GraphBackend, NodeKind
from .networkx_backend import NetworkXBackend

__all__ = [
    "GraphBackend",
    "NodeKind",
    "NetworkXBackend",
]
