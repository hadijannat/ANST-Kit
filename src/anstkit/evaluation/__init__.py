"""Evaluation module for ANST-Kit.

Provides metrics collection, scenario generation, and ablation study infrastructure
for evaluating the triad architecture's effectiveness.
"""

from .ablations import AblationConfig, AblationResult, AblationRunner, AblationType
from .metrics import PhysicsMetrics, StructuralMetrics
from .scenarios import Scenario, ScenarioGenerator, ScenarioType

__all__ = [
    "StructuralMetrics",
    "PhysicsMetrics",
    "ScenarioGenerator",
    "ScenarioType",
    "Scenario",
    "AblationType",
    "AblationConfig",
    "AblationResult",
    "AblationRunner",
]
