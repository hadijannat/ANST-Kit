"""Tests for MLflow experiment tracking."""

from unittest.mock import patch


def test_training_tracker_logs_params():
    """Verify tracker logs parameters to MLflow."""
    from anstkit.tracking import TrainingTracker

    with patch("mlflow.log_params") as mock_log:
        with patch("mlflow.set_experiment"):
            with patch("mlflow.start_run"):
                with patch("mlflow.end_run"):
                    tracker = TrainingTracker(experiment_name="test")
                    with tracker:
                        tracker.log_params({"steps": 100, "seed": 42})
                    mock_log.assert_called_once_with({"steps": 100, "seed": 42})


def test_training_tracker_logs_metrics():
    """Verify tracker logs metrics with optional step."""
    from anstkit.tracking import TrainingTracker

    with patch("mlflow.log_metrics") as mock_log:
        with patch("mlflow.set_experiment"):
            with patch("mlflow.start_run"):
                with patch("mlflow.end_run"):
                    tracker = TrainingTracker(experiment_name="test")
                    with tracker:
                        tracker.log_metrics({"loss": 0.01}, step=100)
                    mock_log.assert_called_once_with({"loss": 0.01}, step=100)


def test_training_tracker_logs_artifact():
    """Verify tracker logs artifacts (model files)."""
    from pathlib import Path

    from anstkit.tracking import TrainingTracker

    with patch("mlflow.log_artifact") as mock_log:
        with patch("mlflow.set_experiment"):
            with patch("mlflow.start_run"):
                with patch("mlflow.end_run"):
                    tracker = TrainingTracker(experiment_name="test")
                    with tracker:
                        tracker.log_artifact(Path("/fake/path/model.pt"))
                    mock_log.assert_called_once_with("/fake/path/model.pt")


def test_tracker_works_without_mlflow():
    """Verify graceful degradation when MLflow is not available."""
    # Force MLFLOW_AVAILABLE to False
    import anstkit.tracking as tracking_module
    from anstkit.tracking import TrainingTracker
    original = tracking_module.MLFLOW_AVAILABLE

    try:
        tracking_module.MLFLOW_AVAILABLE = False

        # Should not raise even without MLflow
        tracker = TrainingTracker(experiment_name="test")
        with tracker:
            tracker.log_params({"test": 1})
            tracker.log_metrics({"loss": 0.1})
    finally:
        tracking_module.MLFLOW_AVAILABLE = original
