"""Command line interface for ANST-Kit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .agent_demo import DemoAgent
from .plant_graph import PlantGraph
from .physics_pinn import PhysicsGateConfig, load_pinn, physics_gate, train_pinn
from .schemas import BenchmarkResult, PlantState
from .orchestrator import TriadOrchestrator
from .evaluation.ablations import AblationConfig, AblationRunner, AblationType

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "tank_pinn.pt"


def cmd_run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    agent = DemoAgent(seed=args.seed)
    plant = PlantGraph()

    if not DEFAULT_WEIGHTS.exists():
        print("No PINN weights found; training a small model first...")
        train_pinn(DEFAULT_WEIGHTS, steps=1500, seed=args.seed)

    pinn = load_pinn(DEFAULT_WEIGHTS)

    # Smaller horizon for fast benchmarking
    fast_cfg = PhysicsGateConfig(horizon=2.0, n_eval=32)

    orch = TriadOrchestrator(agent=agent, plant=plant, pinn=pinn, state=PlantState(), physics_cfg=fast_cfg)

    decision = orch.step(args.goal)

    print("=== Decision ===")
    print(f"Approved: {decision.approved}")
    print(f"Final actions: {decision.final_actions}")
    print(f"Structural gate: {decision.structural.status} | reasons={decision.structural.reasons}")
    print(f"Physics gate: {decision.physics.status} | reasons={decision.physics.reasons}")
    print(f"Physics metrics: {decision.physics.metrics}")
    print("=== New State ===")
    print(orch.state)


def cmd_train_pinn(args: argparse.Namespace) -> None:
    out = Path(args.out)
    metrics = train_pinn(out, steps=args.steps, seed=args.seed, lr=args.lr)
    print("Saved weights to", out)
    print("Final metrics:", metrics)


def cmd_benchmark(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    agent = DemoAgent(seed=args.seed)
    plant = PlantGraph()

    if not DEFAULT_WEIGHTS.exists():
        train_pinn(DEFAULT_WEIGHTS, steps=1500, seed=args.seed)

    pinn = load_pinn(DEFAULT_WEIGHTS)

    # Smaller horizon for fast benchmarking
    fast_cfg = PhysicsGateConfig(horizon=2.0, n_eval=32)

    unsafe_baseline = 0
    unsafe_triad = 0
    approved_triad = 0

    for i in range(args.n):
        state = PlantState(
            tank_level=float(rng.uniform(0.05, 0.95)),
            pump_speed=float(rng.uniform(0.0, 1.0)),
            valve_opening=float(rng.uniform(0.0, 1.0)),
        )

        goal = rng.choice([
            "increase throughput",
            "reduce level",
            "stabilize unit",
            "increase flow",
        ])

        # Baseline: execute first proposal without gates (clip only to avoid numerical issues)
        proposal = agent.propose(goal, state)
        # Unsafe if: unknown target OR out-of-range OR overflow predicted by physics model
        struct_ok = plant.structural_gate(proposal.actions).status.value == "pass"
        phys_ok = True
        if struct_ok:
            phys_ok = physics_gate(pinn, state, proposal.actions, fast_cfg).status.value == "pass"

        if not (struct_ok and phys_ok):
            unsafe_baseline += 1

        # Triad: full orchestrator
        orch = TriadOrchestrator(agent=agent, plant=plant, pinn=pinn, state=state, physics_cfg=fast_cfg)
        decision = orch.step(goal)
        if decision.approved:
            approved_triad += 1
        else:
            # If rejected, count as unsafe prevented (triad unsafe execution = 0 by definition)
            pass

        # For reporting, we treat "triad unsafe" as "did any approved plan violate gates".
        if decision.approved:
            if not (decision.structural.status.value == "pass" and decision.physics.status.value == "pass"):
                unsafe_triad += 1

    res = BenchmarkResult(
        n=args.n,
        baseline_unsafe_rate=unsafe_baseline / args.n,
        triad_unsafe_rate=unsafe_triad / max(approved_triad, 1),
        triad_approval_rate=approved_triad / args.n,
    )

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(res.model_dump(), f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print("=== Benchmark ===")
        print(res.model_dump())


def cmd_ablation(args: argparse.Namespace) -> None:
    """Run ablation study comparing triad configurations."""
    import json
    from dataclasses import asdict

    config = AblationConfig(
        n_trials=args.n,
        seed=args.seed,
    )

    print(f"Running ablation study with {args.n} trials (seed={args.seed})...")
    runner = AblationRunner(config)
    results = runner.run_all()

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)

    for atype, result in results.items():
        print(f"\n{atype.value}:")
        print(f"  Unsafe rate:    {result.unsafe_rate:.3f}")
        print(f"  Approval rate:  {result.approval_rate:.3f}")
        print(f"  Mean latency:   {result.mean_latency_ms:.2f} ms")
        print(f"  Struct vetoes:  {result.structural_veto_rate:.3f}")
        print(f"  Physics vetoes: {result.physics_veto_rate:.3f}")

    if args.output:
        output = {k.value: asdict(v) for k, v in results.items()}
        # Convert AblationType enum to string for JSON serialization
        for v in output.values():
            v["ablation_type"] = v["ablation_type"].value
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="anstkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run one propose→verify→execute step")
    p_run.add_argument("--goal", required=True)
    p_run.add_argument("--seed", type=int, default=7)
    p_run.set_defaults(func=cmd_run)

    p_train = sub.add_parser("train-pinn", help="Train the demo PINN and save weights")
    p_train.add_argument("--out", default=str(DEFAULT_WEIGHTS))
    p_train.add_argument("--steps", type=int, default=3000)
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p_train.set_defaults(func=cmd_train_pinn)

    p_bench = sub.add_parser("benchmark", help="Run a small reproducibility benchmark")
    p_bench.add_argument("--n", type=int, default=500)
    p_bench.add_argument("--seed", type=int, default=7)
    p_bench.add_argument("--output", type=str, help="Output JSON file for results")
    p_bench.set_defaults(func=cmd_benchmark)

    p_ablation = sub.add_parser("ablation", help="Run ablation study comparing triad configurations")
    p_ablation.add_argument("--n", type=int, default=100, help="Number of trials")
    p_ablation.add_argument("--seed", type=int, default=7, help="Random seed")
    p_ablation.add_argument("--output", type=str, help="Output JSON file for results")
    p_ablation.set_defaults(func=cmd_ablation)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
