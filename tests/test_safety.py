"""Unit tests for per-timestep hard-limit safety checks."""

import numpy as np
import pytest

from cotter.tests.safety import (
    SafetyDecision,
    SafetyLimit,
    check_step,
    evaluate_safety,
)

VEL = "joint_velocities"
FORCE = "actuator_forces"
LIMITS = [SafetyLimit(VEL, 10.0), SafetyLimit(FORCE, 3.0)]


def step(vel, force):
    return {VEL: np.asarray(vel, dtype=float), FORCE: np.asarray(force, dtype=float)}


class TestCheckStep:
    def test_within_limits_no_violation(self):
        assert check_step(step([1.0, -2.0], [0.5]), LIMITS) == []

    def test_negative_values_checked_by_magnitude(self):
        v = check_step(step([-10.5, 0.0], [0.0]), LIMITS)
        assert len(v) == 1
        assert v[0].quantity == VEL
        assert v[0].value == -10.5
        assert v[0].element == 0

    def test_exactly_at_limit_passes(self):
        assert check_step(step([10.0], [3.0]), LIMITS) == []

    def test_scalar_quantity(self):
        v = check_step({"contact_count": 2}, [SafetyLimit("contact_count", 1.0)])
        assert len(v) == 1
        assert v[0].element is None

    def test_missing_quantity_raises(self):
        with pytest.raises(KeyError, match="joint_velocities"):
            check_step({"other": 1.0}, LIMITS)

    def test_multiple_violations_in_one_step(self):
        v = check_step(step([11.0, -12.0], [4.0]), LIMITS)
        assert len(v) == 3
        assert {x.quantity for x in v} == {VEL, FORCE}


class TestEvaluateSafety:
    def test_clean_rollouts_pass(self):
        episodes = [[step([1.0], [0.5])] * 5, [step([2.0], [1.0])] * 3]
        res = evaluate_safety(episodes, LIMITS)
        assert res.decision == SafetyDecision.PASS
        assert res.n_trials == 2
        assert res.n_timesteps_checked == 8
        assert res.n_violations == 0
        assert res.worst_observed[VEL] == 2.0
        assert res.worst_observed[FORCE] == 1.0

    def test_single_violation_anywhere_fails(self):
        episodes = [
            [step([1.0], [0.5])] * 10,
            [step([1.0], [0.5])] * 4 + [step([10.1], [0.5])] + [step([1.0], [0.5])] * 5,
        ]
        res = evaluate_safety(episodes, LIMITS)
        assert res.decision == SafetyDecision.FAIL
        assert res.n_violations == 1
        v = res.violations[0]
        assert (v.trial, v.timestep, v.quantity) == (1, 4, VEL)
        assert v.value == pytest.approx(10.1)
        assert v.limit == 10.0

    def test_no_averaging_tiny_excursion_fails(self):
        # 999 perfect steps + one marginal violation must still FAIL
        episodes = [[step([0.0], [0.0])] * 999 + [step([10.000001], [0.0])]]
        assert evaluate_safety(episodes, LIMITS).decision == SafetyDecision.FAIL

    def test_worst_observed_tracked_even_when_passing(self):
        episodes = [[step([3.0], [2.9]), step([-9.9], [0.1])]]
        res = evaluate_safety(episodes, LIMITS)
        assert res.decision == SafetyDecision.PASS
        assert res.worst_observed[VEL] == pytest.approx(9.9)
        assert res.worst_observed[FORCE] == pytest.approx(2.9)

    def test_no_limits_rejected(self):
        with pytest.raises(ValueError):
            evaluate_safety([[step([0.0], [0.0])]], [])

    def test_nonpositive_limit_rejected(self):
        with pytest.raises(ValueError):
            SafetyLimit(VEL, 0.0)

    def test_result_dict(self):
        res = evaluate_safety([[step([11.0], [0.0])]], LIMITS)
        d = res.to_dict()
        assert d["decision"] == "FAIL"
        assert d["n_violations"] == 1
        assert d["violations"][0]["quantity"] == VEL
