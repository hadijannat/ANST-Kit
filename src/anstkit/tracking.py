"""MLflow experiment tracking for ANST-Kit.

Provides a wrapper around MLflow for tracking PINN training experiments,
with graceful degradation when MLflow is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class TrainingTracker:
    """MLflow wrapper for PINN training experiments.

    Usage:
        with TrainingTracker(experiment_name="pinn-training") as tracker:
            tracker.log_params({"steps": 3000, "seed": 7})
            # ... training loop ...
            tracker.log_metrics({"loss": 0.001}, step=3000)
            tracker.log_artifact(Path("models/tank_pinn.pt"))
    """

    experiment_name: str = "anstkit-pinn"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None

    def __post_init__(self) -> None:
        if not MLFLOW_AVAILABLE:
            return
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def __enter__(self) -> "TrainingTracker":
        if MLFLOW_AVAILABLE:
            self._run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, *args: Any) -> None:
        if MLFLOW_AVAILABLE:
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        if MLFLOW_AVAILABLE:
            mlflow.log_params(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow with optional step number."""
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        """Log an artifact (model file, etc.) to MLflow."""
        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(path))


def track_pinn_training(
    out_path: Path,
    steps: int = 3000,
    seed: int = 7,
    lr: float = 3e-4,
    experiment_name: str = "anstkit-pinn",
) -> Dict[str, float]:
    """Train PINN with MLflow tracking.

    Args:
        out_path: Path to save model weights.
        steps: Number of training steps.
        seed: Random seed for reproducibility.
        lr: Learning rate.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary of final training metrics.
    """
    from anstkit.physics_pinn import train_pinn

    with TrainingTracker(experiment_name=experiment_name) as tracker:
        tracker.log_params(
            {
                "steps": steps,
                "seed": seed,
                "lr": lr,
                "model_type": "TankPINN",
            }
        )

        metrics = train_pinn(out_path, steps=steps, seed=seed, lr=lr)

        tracker.log_metrics(metrics)
        tracker.log_artifact(out_path)

    return metrics
