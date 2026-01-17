import pytest
from anstkit.evaluation.ablations import AblationConfig, AblationRunner, AblationType


def test_ablation_runner_executes_all_types():
    config = AblationConfig(n_trials=10, seed=42)
    runner = AblationRunner(config)
    results = runner.run_all()

    assert AblationType.BRAIN_ONLY in results
    assert AblationType.BRAIN_STRUCTURAL in results
    assert AblationType.BRAIN_PHYSICS in results
    assert AblationType.FULL_TRIAD in results

    # Full triad should have lowest unsafe rate (or equal)
    assert results[AblationType.FULL_TRIAD].unsafe_rate <= results[AblationType.BRAIN_ONLY].unsafe_rate


def test_ablation_result_fields():
    config = AblationConfig(n_trials=5, seed=42)
    runner = AblationRunner(config)
    results = runner.run_all()

    for atype, result in results.items():
        assert result.ablation_type == atype
        assert result.n_trials >= 0
        assert 0.0 <= result.unsafe_rate <= 1.0
        assert 0.0 <= result.approval_rate <= 1.0
        assert result.mean_latency_ms >= 0


def test_ablation_runner_deterministic():
    config1 = AblationConfig(n_trials=5, seed=123)
    config2 = AblationConfig(n_trials=5, seed=123)

    runner1 = AblationRunner(config1)
    runner2 = AblationRunner(config2)

    results1 = runner1.run_all()
    results2 = runner2.run_all()

    for atype in AblationType:
        assert results1[atype].unsafe_rate == results2[atype].unsafe_rate
        assert results1[atype].approval_rate == results2[atype].approval_rate
