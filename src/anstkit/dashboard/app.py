"""Streamlit dashboard for ANST-Kit triadic runtime assurance demo.

Run with: streamlit run src/anstkit/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from anstkit.agent_demo import DemoAgent
from anstkit.config import get_settings
from anstkit.orchestrator import OrchestratorConfig, TriadOrchestrator
from anstkit.physics_pinn import (
    PhysicsGateConfig,
    load_pinn,
    simulate_horizon,
    train_pinn,
)
from anstkit.plant_graph import PlantGraph
from anstkit.schemas import PlantState, ValidationStatus

# Assets path for architecture diagram (relative to package root)
ASSETS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "assets"


@st.cache_resource
def load_components(seed: int | None = None):
    """Load and cache the orchestrator components."""
    settings = get_settings()
    seed = seed if seed is not None else settings.seed
    weights_path = settings.pinn_weights_path

    # Train PINN if weights don't exist
    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Training PINN model (first run only)..."):
            train_pinn(weights_path, steps=settings.pinn.training_steps, seed=seed)

    pinn = load_pinn(weights_path)
    agent = DemoAgent(seed=seed)  # Uses config defaults for hallucination rates
    plant = PlantGraph()
    return pinn, agent, plant


def create_trajectory_plot(
    t: np.ndarray,
    h: np.ndarray,
    h_max: float = 1.0,
    title: str = "Predicted Tank Level Trajectory",
) -> go.Figure:
    """Create a Plotly figure showing the predicted trajectory with safety bounds."""
    fig = go.Figure()

    # Safety bounds (shaded regions for overflow/empty)
    fig.add_hrect(
        y0=h_max,
        y1=max(h.max() * 1.1, h_max + 0.1),
        fillcolor="rgba(255, 0, 0, 0.1)",
        line_width=0,
        annotation_text="Overflow Zone",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=min(h.min() * 1.1, -0.1),
        y1=0,
        fillcolor="rgba(255, 0, 0, 0.1)",
        line_width=0,
        annotation_text="Empty Zone",
        annotation_position="bottom left",
    )

    # Safety limit lines
    fig.add_hline(
        y=h_max,
        line_dash="dash",
        line_color="red",
        annotation_text=f"H_MAX = {h_max}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="H_MIN = 0",
        annotation_position="bottom right",
    )

    # Predicted trajectory
    fig.add_trace(
        go.Scatter(
            x=t,
            y=h,
            mode="lines",
            name="Predicted H(t)",
            line=dict(color="blue", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (normalized units)",
        yaxis_title="Tank Level H(t)",
        yaxis=dict(range=[min(h.min() - 0.1, -0.1), max(h.max() + 0.1, h_max + 0.2)]),
        showlegend=True,
        height=400,
    )

    return fig


def display_gate_status(label: str, status: ValidationStatus, reasons: list[str]):
    """Display a gate status with color-coded indicator."""
    if status == ValidationStatus.PASS:
        st.success(f"**{label}:** PASS")
    else:
        st.error(f"**{label}:** FAIL")
        for reason in reasons:
            st.caption(f"  - {reason}")


def main():
    st.set_page_config(
        page_title="ANST-Kit: Triadic Runtime Assurance",
        page_icon="🔒",
        layout="wide",
    )

    # Header with architecture diagram
    st.title("ANST-Kit: Triadic Runtime Assurance Demo")

    arch_diagram = ASSETS_PATH / "architecture_diagram.png"
    if arch_diagram.exists():
        st.image(str(arch_diagram), use_container_width=True)
    else:
        st.info("Architecture diagram not found. Run from the project root directory.")

    st.markdown("---")

    # Load cached components
    pinn, agent, plant = load_components()

    # Sidebar: Inputs
    with st.sidebar:
        st.header("Configuration")

        goal = st.text_input(
            "Goal",
            value="increase throughput",
            help="Natural language goal for the agent (e.g., 'increase throughput', 'reduce level', 'stabilize')",
        )

        st.subheader("Plant State")
        tank_level = st.slider(
            "Tank Level",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Current normalized tank level (0 = empty, 1 = full)",
        )
        pump_speed = st.slider(
            "Pump Speed",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Current pump speed setpoint (0 = off, 1 = max)",
        )
        valve_opening = st.slider(
            "Valve Opening",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Current outlet valve opening (0 = closed, 1 = fully open)",
        )

        st.markdown("---")

        # Random seed for reproducibility (default from config)
        settings = get_settings()
        seed = st.number_input("Random Seed", value=settings.seed, min_value=0, step=1)

        run_button = st.button("Run Proposal", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption(
            "**Demo Agent:** 20% chance of hallucinating invalid targets, "
            "20% chance of proposing out-of-range values."
        )

    # Main area
    if run_button:
        # Create fresh agent with the seed (uses config defaults for hallucination rates)
        settings = get_settings()
        agent = DemoAgent(seed=int(seed))

        state = PlantState(
            tank_level=tank_level,
            pump_speed=pump_speed,
            valve_opening=valve_opening,
        )

        # Run orchestrator with physics config from centralized settings
        physics_cfg = PhysicsGateConfig(
            horizon=settings.physics.horizon,
            n_eval=settings.physics.n_eval,
            h_max=settings.physics.h_max,
        )
        orch = TriadOrchestrator(
            agent=agent,
            plant=plant,
            pinn=pinn,
            state=state,
            cfg=OrchestratorConfig(max_iterations=3),
            physics_cfg=physics_cfg,
        )

        with st.spinner("Running triadic verification..."):
            decision = orch.step(goal)

        # Decision Banner
        st.header("Decision")
        if decision.approved:
            st.success("## APPROVED", icon="✅")
        else:
            st.error("## REJECTED", icon="❌")

        # Gate Status
        st.subheader("Gate Results")
        col1, col2 = st.columns(2)

        with col1:
            display_gate_status(
                "Structural Gate",
                decision.structural.status,
                decision.structural.reasons,
            )

        with col2:
            display_gate_status(
                "Physics Gate",
                decision.physics.status,
                decision.physics.reasons,
            )

        # Trajectory Plot
        st.subheader("Predicted Trajectory")

        # Get proposed setpoints for trajectory visualization
        u_pump = state.pump_speed
        u_valve = state.valve_opening

        for action in decision.final_actions:
            if action.type.value == "set_pump_speed":
                u_pump = action.value
            elif action.type.value == "set_valve_opening":
                u_valve = action.value

        # If no actions approved, use the original proposal's first attempt
        if not decision.final_actions:
            # Re-propose to show what trajectory would have been
            proposal = agent.propose(goal, state)
            for action in proposal.actions:
                if action.type.value == "set_pump_speed":
                    u_pump = action.value
                elif action.type.value == "set_valve_opening":
                    u_valve = action.value

        # Simulate trajectory
        t_arr, h_arr, residual = simulate_horizon(pinn, state, u_pump, u_valve, physics_cfg)

        fig = create_trajectory_plot(t_arr, h_arr, h_max=physics_cfg.h_max)
        st.plotly_chart(fig, use_container_width=True)

        # Physics metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Max H(t)", f"{h_arr.max():.3f}", delta=None)
        with metrics_col2:
            st.metric("Min H(t)", f"{h_arr.min():.3f}", delta=None)
        with metrics_col3:
            st.metric("Residual MSE", f"{residual:.2e}", delta=None)

        # Audit Timeline
        st.subheader("Audit Timeline")

        with st.expander("Proposal Details", expanded=True):
            st.write(f"**Goal:** {goal}")
            st.write(f"**Initial State:** tank_level={state.tank_level:.2f}, pump_speed={state.pump_speed:.2f}, valve_opening={state.valve_opening:.2f}")

            if decision.final_actions:
                st.write("**Approved Actions:**")
                for i, action in enumerate(decision.final_actions):
                    st.code(f"{i+1}. {action.type.value} on {action.target_id} = {action.value:.3f}")
            else:
                st.write("**No actions approved.**")

        with st.expander("Structural Gate Evidence"):
            st.json(decision.structural.evidence if decision.structural.evidence else {"note": "No evidence recorded"})

        with st.expander("Physics Gate Evidence"):
            st.json(decision.physics.evidence if decision.physics.evidence else {"note": "No evidence recorded"})

        with st.expander("Physics Gate Metrics"):
            st.json(decision.physics.metrics)

    else:
        # Initial state - show instructions
        st.info(
            "Configure the plant state in the sidebar and click **Run Proposal** to see the "
            "triadic runtime assurance in action.\n\n"
            "**Try these scenarios:**\n"
            "- Normal: Use defaults and click Run multiple times (20% chance of gate failures)\n"
            "- Overflow risk: Set tank_level=0.95, pump_speed=1.0 → likely physics rejection\n"
            "- Goals: Try 'increase throughput', 'reduce level', or 'stabilize unit'"
        )


if __name__ == "__main__":
    main()
