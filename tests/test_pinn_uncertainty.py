"""Tests for PINN uncertainty quantification via ensemble."""

import pytest

from anstkit.physics_pinn import EnsemblePINN, PINNConfidence
from anstkit.schemas import ActionType, ControlAction, PlantState


def test_ensemble_provides_uncertainty():
    """Ensemble should provide epistemic uncertainty estimate."""
    ensemble = EnsemblePINN(n_models=3, seed=42)
    ensemble.train_all(steps=100)  # Quick training for test

    state = PlantState(tank_level=0.5, pump_speed=0.5, valve_opening=0.5)
    actions = [
        ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.7),
    ]

    confidence = ensemble.predict_with_uncertainty(state, actions)

    assert isinstance(confidence, PINNConfidence)
    assert confidence.epistemic_std >= 0
    assert isinstance(confidence.is_ood, bool)


def test_ood_detection_flags_extreme_inputs():
    """Extreme inputs outside training distribution should have high OOD score."""
    ensemble = EnsemblePINN(n_models=3, seed=42)
    ensemble.train_all(steps=100)

    # Extreme state outside training distribution
    extreme_state = PlantState(tank_level=0.99, pump_speed=1.0, valve_opening=0.0)
    actions = [
        ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=1.0),
    ]

    confidence = ensemble.predict_with_uncertainty(extreme_state, actions)
    # Should flag as potentially OOD due to high uncertainty
    assert confidence.ood_score > 0


def test_ensemble_is_deterministic_with_seed():
    """Same seed should produce same results."""
    state = PlantState(tank_level=0.5, pump_speed=0.5, valve_opening=0.5)
    actions = [
        ControlAction(type=ActionType.SET_PUMP_SPEED, target_id="P1", value=0.6),
    ]

    ensemble1 = EnsemblePINN(n_models=2, seed=123)
    ensemble1.train_all(steps=50)
    conf1 = ensemble1.predict_with_uncertainty(state, actions)

    ensemble2 = EnsemblePINN(n_models=2, seed=123)
    ensemble2.train_all(steps=50)
    conf2 = ensemble2.predict_with_uncertainty(state, actions)

    assert conf1.epistemic_std == pytest.approx(conf2.epistemic_std, rel=1e-3)
    assert conf1.ood_score == pytest.approx(conf2.ood_score, rel=1e-3)


def test_ensemble_requires_training():
    """Calling predict without training should raise."""
    ensemble = EnsemblePINN(n_models=2, seed=42)
    state = PlantState(tank_level=0.5, pump_speed=0.5, valve_opening=0.5)
    actions = []

    with pytest.raises(RuntimeError, match="not trained"):
        ensemble.predict_with_uncertainty(state, actions)
