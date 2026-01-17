"""Ablation study runner for evaluating triad component contributions.

Runs four ablation configurations:
1. BRAIN_ONLY: Agent proposals without any gates (baseline)
2. BRAIN_STRUCTURAL: Agent + structural gate (no physics)
3. BRAIN_PHYSICS: Agent + physics gate (no structural)
4. FULL_TRIAD: Agent + both gates (complete system)

This enables quantifying each gate's contribution to safety.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from anstkit.agent_demo import DemoAgent
from anstkit.evaluation.scenarios import Scenario, ScenarioGenerator, ScenarioType
from anstkit.physics_pinn import PhysicsGateConfig, load_pinn, physics_gate, train_pinn
from anstkit.plant_graph import PlantGraph


class AblationType(str, Enum):
    """Types of ablation configurations."""

    BRAIN_ONLY = "brain_only"
    BRAIN_STRUCTURAL = "brain_structural"
    BRAIN_PHYSICS = "brain_physics"
    FULL_TRIAD = "full_triad"


@dataclass
class AblationConfig:
    """Configuration for ablation study.

    Attributes:
        n_trials: Total number of trials to run.
        seed: Random seed for reproducibility.
        scenario_mix: Distribution of scenario types.
        pinn_weights_path: Path to PINN weights (trains if missing).
    """

    n_trials: int = 100
    seed: int = 42
    scenario_mix: Dict[ScenarioType, float] = field(
        default_factory=lambda: {
            ScenarioType.NORMAL: 0.6,
            ScenarioType.EDGE: 0.3,
            ScenarioType.ADVERSARIAL: 0.1,
        }
    )
    pinn_weights_path: Optional[Path] = None


@dataclass
class AblationResult:
    """Results from one ablation configuration.

    Attributes:
        ablation_type: Which gates were enabled.
        n_trials: Number of trials executed.
        unsafe_rate: Fraction of approved actions that would be unsafe.
        approval_rate: Fraction of proposals that were approved.
        mean_latency_ms: Average validation latency.
        structural_veto_rate: Fraction vetoed by structural gate.
        physics_veto_rate: Fraction vetoed by physics gate.
    """

    ablation_type: AblationType
    n_trials: int
    unsafe_rate: float
    approval_rate: float
    mean_latency_ms: float
    structural_veto_rate: float
    physics_veto_rate: float


class AblationRunner:
    """Run ablation studies comparing triad configurations.

    Executes the same scenarios across all ablation types to enable
    fair comparison of gate contributions to safety.
    """

    DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent.parent / "models" / "tank_pinn.pt"

    def __init__(self, config: AblationConfig):
        """Initialize ablation runner.

        Args:
            config: Ablation study configuration.
        """
        self.config = config
        self.scenario_gen = ScenarioGenerator(seed=config.seed)
        self.agent = DemoAgent(seed=config.seed)
        self.plant = PlantGraph()
        self.physics_cfg = PhysicsGateConfig(horizon=2.0, n_eval=32)

        # Load or train PINN
        weights_path = config.pinn_weights_path or self.DEFAULT_WEIGHTS
        if not weights_path.exists():
            print(f"Training PINN (weights not found at {weights_path})...")
            train_pinn(weights_path, steps=1500, seed=config.seed)
        self.pinn = load_pinn(weights_path)

    def run_all(self) -> Dict[AblationType, AblationResult]:
        """Run all ablation configurations.

        Returns:
            Dict mapping ablation type to results.
        """
        scenarios = self._generate_scenarios()
        results = {}
        for ablation_type in AblationType:
            results[ablation_type] = self._run_ablation(ablation_type, scenarios)
        return results

    def _generate_scenarios(self) -> List[Scenario]:
        """Generate mixed scenarios according to config."""
        scenarios = []
        for stype, fraction in self.config.scenario_mix.items():
            n = max(1, int(self.config.n_trials * fraction))
            scenarios.extend(self.scenario_gen.generate(n, stype))
        return scenarios[: self.config.n_trials]  # Ensure exact count

    def _run_ablation(
        self, ablation_type: AblationType, scenarios: List[Scenario]
    ) -> AblationResult:
        """Run one ablation configuration across all scenarios.

        Args:
            ablation_type: Which gates to enable.
            scenarios: List of test scenarios.

        Returns:
            AblationResult with aggregated metrics.
        """
        unsafe_count = 0
        approved_count = 0
        structural_vetos = 0
        physics_vetos = 0
        latencies = []

        for scenario in scenarios:
            start = time.perf_counter()
            proposal = self.agent.propose(scenario.goal, scenario.state)

            structural_pass = True
            physics_pass = True

            # Apply gates based on ablation type
            if ablation_type in [AblationType.BRAIN_STRUCTURAL, AblationType.FULL_TRIAD]:
                struct_result = self.plant.structural_gate(proposal.actions)
                structural_pass = struct_result.status.value == "pass"
                if not structural_pass:
                    structural_vetos += 1

            if ablation_type in [AblationType.BRAIN_PHYSICS, AblationType.FULL_TRIAD]:
                if structural_pass:  # Only check physics if structure passed
                    phys_result = physics_gate(
                        self.pinn, scenario.state, proposal.actions, self.physics_cfg
                    )
                    physics_pass = phys_result.status.value == "pass"
                    if not physics_pass:
                        physics_vetos += 1

            approved = structural_pass and physics_pass
            if approved:
                approved_count += 1

            # Check if approved action would actually be unsafe
            # (approved an action on a scenario marked as unsafe)
            if approved and not scenario.expected_safe:
                unsafe_count += 1

            latencies.append((time.perf_counter() - start) * 1000)

        n = len(scenarios)
        return AblationResult(
            ablation_type=ablation_type,
            n_trials=n,
            unsafe_rate=unsafe_count / n if n > 0 else 0.0,
            approval_rate=approved_count / n if n > 0 else 0.0,
            mean_latency_ms=sum(latencies) / n if n > 0 else 0.0,
            structural_veto_rate=structural_vetos / n if n > 0 else 0.0,
            physics_veto_rate=physics_vetos / n if n > 0 else 0.0,
        )
