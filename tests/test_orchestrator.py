from pathlib import Path

from anstkit.agent_demo import DemoAgent
from anstkit.orchestrator import TriadOrchestrator
from anstkit.physics_pinn import PhysicsGateConfig, load_pinn
from anstkit.plant_graph import PlantGraph
from anstkit.schemas import PlantState


def test_orchestrator_never_approves_invalid_plan():
    weights = Path(__file__).resolve().parents[1] / "models" / "tank_pinn.pt"
    pinn = load_pinn(weights)

    agent = DemoAgent(seed=42)
    plant = PlantGraph()

    fast_cfg = PhysicsGateConfig(horizon=2.0, n_eval=16, residual_threshold=1e-2)
    orch = TriadOrchestrator(agent=agent, plant=plant, pinn=pinn, state=PlantState(), physics_cfg=fast_cfg)

    for _ in range(2):
        decision = orch.step("increase throughput")
        if decision.approved:
            assert decision.structural.status.value == "pass"
            assert decision.physics.status.value == "pass"
