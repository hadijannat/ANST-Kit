"""Metrics dataclasses for evaluation results.

These metrics capture the key performance indicators for both the structural
(graph-based) and physics (PINN-based) validation gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class StructuralMetrics:
    """Aggregated metrics from structural gate evaluation.

    Tracks rates of different failure modes:
    - unknown_target: Agent referenced non-existent asset
    - wrong_type: Action applied to incompatible asset type
    - disconnected: Asset not connected to controlled unit
    """

    unknown_target_rate: float
    wrong_type_rate: float
    disconnected_rate: float
    total_proposals: int

    @classmethod
    def from_results(cls, results: List[Dict[str, Any]]) -> "StructuralMetrics":
        """Compute metrics from a list of validation results.

        Args:
            results: List of dicts with boolean keys for each failure mode.

        Returns:
            StructuralMetrics with computed rates.
        """
        n = len(results)
        if n == 0:
            return cls(0.0, 0.0, 0.0, 0)
        return cls(
            unknown_target_rate=sum(1 for r in results if r.get("unknown_target")) / n,
            wrong_type_rate=sum(1 for r in results if r.get("wrong_type")) / n,
            disconnected_rate=sum(1 for r in results if r.get("disconnected")) / n,
            total_proposals=n,
        )


@dataclass
class PhysicsMetrics:
    """Aggregated metrics from physics gate evaluation.

    Tracks rates of different physics violations and inference performance:
    - overflow: Predicted tank level exceeds max
    - underflow: Predicted tank level goes negative
    - residual_ood: Physics residual indicates out-of-distribution
    - latency: Inference time for physics predictions
    """

    overflow_rate: float
    underflow_rate: float
    residual_ood_rate: float
    mean_inference_latency_ms: float
    total_checks: int

    @classmethod
    def from_results(cls, results: List[Dict[str, Any]]) -> "PhysicsMetrics":
        """Compute metrics from a list of physics check results.

        Args:
            results: List of dicts with boolean flags and latency_ms values.

        Returns:
            PhysicsMetrics with computed rates and mean latency.
        """
        n = len(results)
        if n == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0)
        latencies = [r.get("latency_ms", 0) for r in results]
        return cls(
            overflow_rate=sum(1 for r in results if r.get("overflow")) / n,
            underflow_rate=sum(1 for r in results if r.get("underflow")) / n,
            residual_ood_rate=sum(1 for r in results if r.get("residual_ood")) / n,
            mean_inference_latency_ms=sum(latencies) / n if latencies else 0.0,
            total_checks=n,
        )
