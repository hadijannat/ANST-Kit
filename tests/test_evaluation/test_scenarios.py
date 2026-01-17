import pytest
from anstkit.evaluation.scenarios import ScenarioGenerator, ScenarioType, Scenario
from anstkit.schemas import PlantState


def test_scenario_generator_produces_valid_states():
    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate(n=10, scenario_type=ScenarioType.NORMAL)
    assert len(scenarios) == 10
    for s in scenarios:
        assert isinstance(s.state, PlantState)
        assert 0 <= s.state.tank_level <= 1
        assert s.scenario_type == ScenarioType.NORMAL


def test_edge_scenarios_at_limits():
    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate(n=5, scenario_type=ScenarioType.EDGE)
    # Edge scenarios should have state values near 0 or 1
    for s in scenarios:
        level = s.state.tank_level
        assert level < 0.1 or level > 0.9


def test_adversarial_scenarios_have_dangerous_combos():
    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate(n=10, scenario_type=ScenarioType.ADVERSARIAL)
    for s in scenarios:
        assert s.scenario_type == ScenarioType.ADVERSARIAL
        # Adversarial scenarios should be marked as expected_safe=False
        assert s.expected_safe is False


def test_scenario_generator_deterministic_with_seed():
    gen1 = ScenarioGenerator(seed=123)
    gen2 = ScenarioGenerator(seed=123)
    scenarios1 = gen1.generate(n=5, scenario_type=ScenarioType.NORMAL)
    scenarios2 = gen2.generate(n=5, scenario_type=ScenarioType.NORMAL)
    for s1, s2 in zip(scenarios1, scenarios2):
        assert s1.state.tank_level == s2.state.tank_level
        assert s1.goal == s2.goal


def test_scenario_includes_goal():
    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate(n=3, scenario_type=ScenarioType.NORMAL)
    for s in scenarios:
        assert isinstance(s.goal, str)
        assert len(s.goal) > 0
