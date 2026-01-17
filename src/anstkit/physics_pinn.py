"""Physics gate implemented as a small parametric PINN.

This is a *toy* demonstration that still preserves the core ANST principle:

- The agent is untrusted.
- The physics model is trusted to enforce a safety envelope.

System modeled (normalized units):

    dH/dt = (k_in * u_pump - k_out * u_valve * sqrt(H)) / A

Where:
- H(t) is normalized tank level in [0, 1]
- u_pump and u_valve are normalized setpoints in [0, 1]
- k_in, k_out, A are fixed constants

The PINN learns H(t) for many parameter tuples (H0, u_pump, u_valve).
At runtime, we infer H(t) over a horizon and veto if:
- max(H) > H_MAX
- min(H) < 0

A residual metric is also computed (mean PDE residual) and can be used as
an out-of-distribution / confidence heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .schemas import ActionType, ControlAction, GateResult, PlantState, ValidationStatus


@dataclass
class TankSystemParams:
    k_in: float = 1.0
    k_out: float = 0.6
    area: float = 1.0
    eps: float = 1e-6


@dataclass
class PhysicsGateConfig:
    horizon: float = 5.0
    n_eval: int = 64
    h_max: float = 1.0
    residual_threshold: float = 5e-3


class TankPINN(nn.Module):
    """Parametric PINN mapping (t, H0, u_pump, u_valve) -> H(t)."""

    def __init__(self, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers = []
        in_dim = 4
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, t: torch.Tensor, h0: torch.Tensor, u_pump: torch.Tensor, u_valve: torch.Tensor) -> torch.Tensor:
        x = torch.cat([t, h0, u_pump, u_valve], dim=1)
        raw = self.net(x)
        # enforce H >= 0; the system constraints check upper bound separately
        return self.softplus(raw)


def _sample_batch(n: int, device: torch.device, h0_range=(0.05, 0.95)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample parameter tuples: (H0, u_pump, u_valve)."""
    h0 = torch.rand(n, 1, device=device) * (h0_range[1] - h0_range[0]) + h0_range[0]
    u_pump = torch.rand(n, 1, device=device)
    u_valve = torch.rand(n, 1, device=device)
    return h0, u_pump, u_valve


def _residual(
    model: TankPINN,
    t: torch.Tensor,
    h0: torch.Tensor,
    u_pump: torch.Tensor,
    u_valve: torch.Tensor,
    sys: TankSystemParams,
) -> torch.Tensor:
    """Compute PDE/ODE residual r(t) = dH/dt - f(H,t,params)."""
    t.requires_grad_(True)
    h = model(t, h0, u_pump, u_valve)

    # derivative dH/dt
    (dh_dt,) = torch.autograd.grad(h.sum(), t, create_graph=True)

    rhs = (sys.k_in * u_pump - sys.k_out * u_valve * torch.sqrt(h + sys.eps)) / sys.area
    r = dh_dt - rhs
    return r


def train_pinn(
    out_path: Path,
    steps: int = 3000,
    seed: int = 7,
    lr: float = 3e-4,
    device: str | None = None,
) -> Dict[str, float]:
    """Train and save a demo PINN.

    The training is intentionally lightweight.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make training cost predictable in small CPU environments.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = TankPINN().to(dev)
    sys = TankSystemParams()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        # Collocation points
        n_col = 64
        t = torch.rand(n_col, 1, device=dev) * 5.0
        h0, u_pump, u_valve = _sample_batch(n_col, dev)

        r = _residual(model, t, h0, u_pump, u_valve, sys)
        loss_phys = (r ** 2).mean()

        # Initial condition loss H(0)=H0
        t0 = torch.zeros(n_col, 1, device=dev)
        h_pred0 = model(t0, h0, u_pump, u_valve)
        loss_ic = ((h_pred0 - h0) ** 2).mean()

        loss = loss_phys + 5.0 * loss_ic
        loss.backward()
        opt.step()

        # Cheap progress prints every 500 steps
        if step % 500 == 0:
            print(f"step={step} loss={loss.item():.6f} phys={loss_phys.item():.6f} ic={loss_ic.item():.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed}, out_path)

    return {"loss": float(loss.item()), "loss_phys": float(loss_phys.item()), "loss_ic": float(loss_ic.item())}


def load_pinn(weights_path: Path, device: str | None = None) -> TankPINN:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(weights_path, map_location=dev)
    model = TankPINN().to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def simulate_horizon(
    model: TankPINN,
    state: PlantState,
    u_pump: float,
    u_valve: float,
    cfg: PhysicsGateConfig,
    sys: TankSystemParams | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simulate H(t) over horizon and return (t, H, residual_mse).

    NOTE: For runtime speed, we compute an *approximate* residual using
    finite differences on the predicted trajectory instead of autograd.
    This keeps the guardrail fast (PINN inference is meant to be ms-level).
    """
    sys = sys or TankSystemParams()
    dev = next(model.parameters()).device

    t = torch.linspace(0.0, cfg.horizon, cfg.n_eval, device=dev).reshape(-1, 1)
    h0 = torch.full_like(t, float(state.tank_level))
    up = torch.full_like(t, float(u_pump))
    uv = torch.full_like(t, float(u_valve))

    with torch.no_grad():
        h = model(t, h0, up, uv).detach().cpu().numpy().reshape(-1)
        t_np = t.detach().cpu().numpy().reshape(-1)

    # Finite-difference residual
    dh_dt = np.gradient(h, t_np)
    rhs = (sys.k_in * float(u_pump) - sys.k_out * float(u_valve) * np.sqrt(h + sys.eps)) / sys.area
    r = dh_dt - rhs
    residual_mse = float(np.mean(r ** 2))

    return t_np, h, residual_mse


def physics_gate(
    model: TankPINN,
    state: PlantState,
    actions: list[ControlAction],
    cfg: PhysicsGateConfig | None = None,
) -> GateResult:
    cfg = cfg or PhysicsGateConfig()

    # Apply actions to compute proposed setpoints (simple: last write wins)
    u_pump = state.pump_speed
    u_valve = state.valve_opening

    for a in actions:
        if a.type == ActionType.SET_PUMP_SPEED:
            u_pump = a.value
        elif a.type == ActionType.SET_VALVE_OPENING:
            u_valve = a.value

    t, h, residual_mse = simulate_horizon(model, state, u_pump, u_valve, cfg)

    reasons = []
    if float(h.max()) > cfg.h_max + 1e-9:
        reasons.append(f"Predicted tank level exceeds limit: max(H)={h.max():.3f} > H_MAX={cfg.h_max:.3f}.")
    if float(h.min()) < -1e-9:
        reasons.append(f"Predicted tank level below 0: min(H)={h.min():.3f}.")
    if residual_mse > cfg.residual_threshold:
        reasons.append(
            f"Physics residual too high: residual_mse={residual_mse:.4e} > threshold={cfg.residual_threshold:.4e}.")

    metrics = {
        "u_pump": float(u_pump),
        "u_valve": float(u_valve),
        "h_max_pred": float(h.max()),
        "h_min_pred": float(h.min()),
        "residual_mse": float(residual_mse),
        "horizon": cfg.horizon,
        "n_eval": cfg.n_eval,
    }

    return GateResult(
        status=ValidationStatus.PASS if not reasons else ValidationStatus.FAIL,
        reasons=reasons,
        metrics=metrics,
    )


def apply_actions_to_state(state: PlantState, actions: list[ControlAction]) -> PlantState:
    """Pure state transition for the demo."""
    new_state = state.model_copy(deep=True)
    for a in actions:
        if a.type == ActionType.SET_PUMP_SPEED:
            new_state.pump_speed = float(a.value)
        elif a.type == ActionType.SET_VALVE_OPENING:
            new_state.valve_opening = float(a.value)

    # Update level by a single Euler step using the same dynamics (for demo).
    sys = TankSystemParams()
    dt = 0.25
    dh = (sys.k_in * new_state.pump_speed - sys.k_out * new_state.valve_opening * np.sqrt(max(new_state.tank_level, 0.0) + sys.eps)) / sys.area
    new_state.tank_level = float(np.clip(new_state.tank_level + dt * dh, 0.0, 1.5))

    return new_state


# ============================================================================
# Uncertainty Quantification via Ensemble
# ============================================================================


@dataclass
class OODConfig:
    """Configuration for out-of-distribution detection.

    Attributes:
        residual_threshold: Max acceptable physics residual MSE.
        ensemble_disagreement_threshold: Max acceptable epistemic std.
        input_distance_threshold: Max Mahalanobis distance from training.
    """

    residual_threshold: float = 5e-3
    ensemble_disagreement_threshold: float = 0.1
    input_distance_threshold: float = 2.0


@dataclass
class PINNConfidence:
    """Confidence estimate from ensemble prediction.

    Attributes:
        prediction: Mean trajectory prediction (H values over horizon).
        epistemic_std: Standard deviation across ensemble members.
        residual_mse: Mean physics residual (should be low for valid predictions).
        is_ood: Whether input is flagged as out-of-distribution.
        ood_score: Normalized OOD score (>1.0 means OOD).
    """

    prediction: np.ndarray
    epistemic_std: float
    residual_mse: float
    is_ood: bool
    ood_score: float


class EnsemblePINN:
    """Ensemble of TankPINNs for uncertainty quantification.

    Uses deep ensemble approach: train multiple PINNs with different
    random initializations, then use disagreement as epistemic uncertainty.
    """

    def __init__(
        self,
        n_models: int = 5,
        seed: int = 42,
        ood_config: Optional[OODConfig] = None,
    ):
        """Initialize ensemble.

        Args:
            n_models: Number of ensemble members.
            seed: Base random seed (each model gets seed + i).
            ood_config: Configuration for OOD detection thresholds.
        """
        self.n_models = n_models
        self.seed = seed
        self.ood_config = ood_config or OODConfig()
        self.models: List[TankPINN] = []
        self._training_mean: Optional[np.ndarray] = None
        self._training_cov_inv: Optional[np.ndarray] = None

    def train_all(self, steps: int = 3000) -> None:
        """Train ensemble with different random seeds.

        Args:
            steps: Training steps per model.
        """
        self.models = []
        for i in range(self.n_models):
            torch.manual_seed(self.seed + i)
            np.random.seed(self.seed + i)
            model = TankPINN()
            # Train inline (similar to train_pinn but without saving)
            self._train_single(model, steps, self.seed + i)
            # Set model to inference mode (PyTorch method, not Python eval)
            model.train(False)
            self.models.append(model)
        # Store training distribution for OOD detection
        self._compute_training_distribution()

    def _train_single(self, model: TankPINN, steps: int, seed: int) -> None:
        """Train a single PINN model."""
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        dev = torch.device("cpu")
        model.to(dev)
        sys = TankSystemParams()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            n_col = 64
            t = torch.rand(n_col, 1, device=dev) * 5.0
            h0, u_pump, u_valve = _sample_batch(n_col, dev)
            r = _residual(model, t, h0, u_pump, u_valve, sys)
            loss_phys = (r**2).mean()

            t0 = torch.zeros(n_col, 1, device=dev)
            h_pred0 = model(t0, h0, u_pump, u_valve)
            loss_ic = ((h_pred0 - h0) ** 2).mean()

            loss = loss_phys + 5.0 * loss_ic
            loss.backward()
            opt.step()

    def _compute_training_distribution(self) -> None:
        """Compute training input statistics for Mahalanobis distance."""
        n_samples = 1000
        rng = np.random.default_rng(self.seed)
        inputs = np.column_stack(
            [
                rng.uniform(0.05, 0.95, n_samples),  # H0 (training range)
                rng.uniform(0, 1, n_samples),  # u_pump
                rng.uniform(0, 1, n_samples),  # u_valve
            ]
        )
        self._training_mean = inputs.mean(axis=0)
        cov = np.cov(inputs.T) + 1e-6 * np.eye(3)
        self._training_cov_inv = np.linalg.pinv(cov)

    def _mahalanobis_distance(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance from training distribution."""
        if self._training_mean is None or self._training_cov_inv is None:
            return 0.0
        diff = x - self._training_mean
        return float(np.sqrt(diff @ self._training_cov_inv @ diff))

    def predict_with_uncertainty(
        self,
        state: PlantState,
        actions: List[ControlAction],
        horizon: float = 5.0,
        n_eval: int = 64,
    ) -> PINNConfidence:
        """Predict trajectory with uncertainty estimation.

        Args:
            state: Current plant state.
            actions: Proposed control actions.
            horizon: Prediction horizon in time units.
            n_eval: Number of evaluation points.

        Returns:
            PINNConfidence with prediction and uncertainty metrics.

        Raises:
            RuntimeError: If ensemble not trained.
        """
        if not self.models:
            raise RuntimeError("Ensemble not trained. Call train_all() first.")

        # Extract proposed setpoints from actions
        u_pump = state.pump_speed
        u_valve = state.valve_opening
        for a in actions:
            if a.type == ActionType.SET_PUMP_SPEED:
                u_pump = a.value
            elif a.type == ActionType.SET_VALVE_OPENING:
                u_valve = a.value

        # Get predictions from all models
        cfg = PhysicsGateConfig(horizon=horizon, n_eval=n_eval)
        predictions = []
        residuals = []

        for model in self.models:
            t_np, h, residual_mse = simulate_horizon(model, state, u_pump, u_valve, cfg)
            predictions.append(h)
            residuals.append(residual_mse)

        predictions_arr = np.array(predictions)
        mean_pred = predictions_arr.mean(axis=0)
        epistemic_std = float(predictions_arr.std(axis=0).mean())
        mean_residual = float(np.mean(residuals))

        # OOD detection via Mahalanobis distance
        input_vec = np.array([state.tank_level, u_pump, u_valve])
        mahal_dist = self._mahalanobis_distance(input_vec)

        # Compute OOD score as max of normalized metrics
        ood_score = max(
            epistemic_std / self.ood_config.ensemble_disagreement_threshold,
            mean_residual / self.ood_config.residual_threshold,
            mahal_dist / self.ood_config.input_distance_threshold,
        )
        is_ood = ood_score > 1.0

        return PINNConfidence(
            prediction=mean_pred,
            epistemic_std=epistemic_std,
            residual_mse=mean_residual,
            is_ood=is_ood,
            ood_score=ood_score,
        )
