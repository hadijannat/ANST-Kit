# ANST-Kit (Agentic Neuro-Symbolic Twin Toolkit)

ANST-Kit is a **reference implementation** of the *Agentic Neuro‑Symbolic Twin* “Triad Architecture”:

- **Brain**: an agent produces action proposals (LLM integration is optional; a deterministic demo agent is included)
- **Map**: a **plant knowledge graph** (GraphRAG-style structural grounding)
- **Guardrail**: a **physics-informed surrogate** (PINN) that vetoes physically unsafe proposals

This repository is designed to be:

- **Reproducible** (seeded training, deterministic demo runs)
- **Publish-ready** (clean experiments, metrics, and an evaluation harness)
- **Extensible** (replace the demo agent with a real LLM, replace the toy PINN with asset-grade PINNs)

> Safety note: this repo is **not** intended to connect to real industrial control systems. It is a research and engineering scaffold.

## Quickstart

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Run the end-to-end demo (agent → graph gate → physics gate)

```bash
python -m anstkit run --goal "increase throughput" --seed 7
```

### 3) Re-train the demo PINN (optional)

```bash
python -m anstkit train-pinn --steps 3000 --seed 7
```

### 4) Run the benchmark

```bash
python -m anstkit benchmark --n 500 --seed 7
```

The benchmark prints:

- baseline unsafe execution rate (no gates)
- triad unsafe execution rate (graph + physics gates)
- triad approval rate

## Repository structure

- `src/anstkit/`
  - `schemas.py` – typed state/action/decision schemas
  - `plant_graph.py` – in-memory plant graph + structural checks
  - `physics_pinn.py` – a small parametric PINN for a tank level ODE
  - `agent_demo.py` – a deterministic agent that intentionally produces invalid proposals sometimes
  - `orchestrator.py` – propose → validate → execute loop
  - `__main__.py` – CLI
- `models/` – saved weights for the demo PINN
- `examples/` – example plant and scenarios
- `tests/` – unit tests

## Extending ANST-Kit

- Replace `agent_demo.py` with a real agent (ReAct / Plan-and-Solve style) and enforce JSON schemas.
- Replace `plant_graph.py` with a graph database backend (Neo4j, RDF) and implement GraphRAG retrieval.
- Replace `physics_pinn.py` with asset-grade PINNs (DeepXDE / PhysicsNeMo / in-house PDEs), and implement fast inference.

## License

MIT License (see `LICENSE`).
