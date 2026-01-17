import pytest

from anstkit.evaluation.metrics import PhysicsMetrics, StructuralMetrics


def test_structural_metrics_computes_rates():
    results = [
        {"unknown_target": True, "wrong_type": False, "disconnected": False},
        {"unknown_target": False, "wrong_type": True, "disconnected": False},
        {"unknown_target": False, "wrong_type": False, "disconnected": False},
    ]
    metrics = StructuralMetrics.from_results(results)
    assert metrics.unknown_target_rate == pytest.approx(1 / 3)
    assert metrics.wrong_type_rate == pytest.approx(1 / 3)
    assert metrics.disconnected_rate == pytest.approx(0.0)


def test_structural_metrics_empty_results():
    metrics = StructuralMetrics.from_results([])
    assert metrics.unknown_target_rate == 0.0
    assert metrics.wrong_type_rate == 0.0
    assert metrics.disconnected_rate == 0.0
    assert metrics.total_proposals == 0


def test_physics_metrics_computes_rates():
    results = [
        {
            "overflow": True,
            "underflow": False,
            "residual_ood": False,
            "uncertainty_ood": False,
            "latency_ms": 10.0,
        },
        {
            "overflow": False,
            "underflow": True,
            "residual_ood": False,
            "uncertainty_ood": True,
            "latency_ms": 20.0,
        },
        {
            "overflow": False,
            "underflow": False,
            "residual_ood": True,
            "uncertainty_ood": False,
            "latency_ms": 15.0,
        },
    ]
    metrics = PhysicsMetrics.from_results(results)
    assert metrics.overflow_rate == pytest.approx(1 / 3)
    assert metrics.underflow_rate == pytest.approx(1 / 3)
    assert metrics.residual_ood_rate == pytest.approx(1 / 3)
    assert metrics.uncertainty_ood_rate == pytest.approx(1 / 3)
    assert metrics.mean_inference_latency_ms == pytest.approx(15.0)
    assert metrics.total_checks == 3


def test_physics_metrics_empty_results():
    metrics = PhysicsMetrics.from_results([])
    assert metrics.overflow_rate == 0.0
    assert metrics.underflow_rate == 0.0
    assert metrics.residual_ood_rate == 0.0
    assert metrics.uncertainty_ood_rate == 0.0
    assert metrics.mean_inference_latency_ms == 0.0
    assert metrics.total_checks == 0
