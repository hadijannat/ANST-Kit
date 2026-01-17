"""Plant graph and structural validation.

This module intentionally uses an **in-memory property graph** (NetworkX)
so the demo runs without external infrastructure.

In a production implementation you would replace this with:
- Neo4j / Memgraph / TigerGraph (property graphs)
- RDF triplestore + SPARQL (semantic graphs)

The key point: the *structural truth* is enforced by graph constraints, so the
agent cannot hallucinate assets or connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Type

import networkx as nx

from .schemas import ActionType, ControlAction, GateResult, ValidationStatus
from .graph.base import GraphBackend
from .graph.networkx_backend import NetworkXBackend
from .ingestion.dexpi_parser import DEXPIParseResult
from .ingestion.topology_extractor import TopologyExtractor


@dataclass
class DemoPlantConfig:
    tank_id: str = "T1"
    pump_id: str = "P1"
    inlet_valve_id: str = "V1"
    outlet_valve_id: str = "V2"


class PlantGraph:
    """In-memory plant knowledge graph with simple validation rules."""

    def __init__(self, cfg: DemoPlantConfig | None = None):
        self.cfg = cfg or DemoPlantConfig()
        self.g = nx.DiGraph()
        self._build_demo_graph()

    def _build_demo_graph(self) -> None:
        # Nodes carry a minimal "type".
        self.g.add_node(self.cfg.pump_id, kind="pump")
        self.g.add_node(self.cfg.inlet_valve_id, kind="valve")
        self.g.add_node(self.cfg.tank_id, kind="tank")
        self.g.add_node(self.cfg.outlet_valve_id, kind="valve")
        self.g.add_node("SINK", kind="sink")

        # Directed edges approximate flow direction.
        self.g.add_edge(self.cfg.pump_id, self.cfg.inlet_valve_id, kind="flows_to")
        self.g.add_edge(self.cfg.inlet_valve_id, self.cfg.tank_id, kind="flows_to")
        self.g.add_edge(self.cfg.tank_id, self.cfg.outlet_valve_id, kind="flows_to")
        self.g.add_edge(self.cfg.outlet_valve_id, "SINK", kind="flows_to")

    def validate_action(self, action: ControlAction) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if action.target_id not in self.g:
            reasons.append(f"Unknown target_id '{action.target_id}'.")
            return False, reasons

        kind = self.g.nodes[action.target_id].get("kind")

        if action.type == ActionType.SET_PUMP_SPEED and kind != "pump":
            reasons.append(f"Action {action.type} not allowed on kind='{kind}'.")

        if action.type == ActionType.SET_VALVE_OPENING and kind != "valve":
            reasons.append(f"Action {action.type} not allowed on kind='{kind}'.")

        # Range checks (structural gate catches obvious command formatting errors).
        if not (0.0 <= action.value <= 1.0):
            reasons.append(f"Value {action.value:.3f} out of allowed range [0, 1].")

        # Connectivity checks: all actuators must be in the same unit as the tank.
        # Here: actuator must have a directed path to the tank OR from the tank.
        if kind in {"pump", "valve"}:
            tank = self.cfg.tank_id
            has_path_to_tank = nx.has_path(self.g, action.target_id, tank)
            has_path_from_tank = nx.has_path(self.g, tank, action.target_id)
            if not (has_path_to_tank or has_path_from_tank):
                reasons.append("Actuator not connected to the controlled unit (tank).")

        ok = len(reasons) == 0
        return ok, reasons

    def structural_gate(self, actions: List[ControlAction]) -> GateResult:
        all_reasons: List[str] = []
        evidence: List[dict] = []
        for a in actions:
            ok, reasons = self.validate_action(a)
            evidence.append(
                {
                    "action": a.model_dump(),
                    "ok": ok,
                    "reasons": reasons,
                }
            )
            if not ok:
                all_reasons.extend(reasons)

        return GateResult(
            status=ValidationStatus.PASS if not all_reasons else ValidationStatus.FAIL,
            reasons=all_reasons,
            metrics={"n_actions": len(actions)},
            evidence=evidence,
        )


class GraphPlant:
    """Plant wrapper around a GraphBackend (NetworkX, GraphRAG, etc.)."""

    def __init__(self, backend: GraphBackend, reference_node: str):
        self.backend = backend
        self.reference_node = reference_node

    @classmethod
    def from_dexpi(
        cls,
        dexpi_data: DEXPIParseResult,
        backend_class: Type[GraphBackend] = NetworkXBackend,
    ) -> "GraphPlant":
        extractor = TopologyExtractor()
        backend = extractor.to_graph_backend(dexpi_data, backend_class)
        reference = extractor.get_tank_reference_node(backend)
        if not reference:
            raise ValueError("No tank reference node found in DEXPI data.")
        return cls(backend=backend, reference_node=reference)

    def structural_gate(self, actions: List[ControlAction]) -> GateResult:
        result = self.backend.structural_gate(actions, self.reference_node)
        evidence: List[dict] = []

        if hasattr(self.backend, "get_structural_evidence"):
            for action in actions:
                evidence.append(
                    {
                        "action": action.model_dump(),
                        "evidence": self.backend.get_structural_evidence(
                            action.target_id, self.reference_node
                        ),
                    }
                )

        if evidence:
            return GateResult(
                status=result.status,
                reasons=result.reasons,
                metrics=result.metrics,
                evidence=evidence,
            )

        return result
