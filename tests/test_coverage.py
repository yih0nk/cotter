"""Tests for operating-envelope coverage sweeps (cotter.coverage)."""

import numpy as np
import pytest

from cotter.coverage import CoverageResult, sobol_scenarios, sweep


class TestSobolScenarios:
    def test_shape_and_bounds(self):
        pts = sobol_scenarios([0, -1], [10, 1], 16, seed=0)
        assert pts.shape == (16, 2)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 10)
        assert np.all(pts[:, 1] >= -1) and np.all(pts[:, 1] <= 1)

    def test_deterministic_for_seed(self):
        a = sobol_scenarios([0], [1], 8, seed=3)
        b = sobol_scenarios([0], [1], 8, seed=3)
        assert np.array_equal(a, b)

    def test_low_discrepancy_spread(self):
        # Sobol should spread across the range, not clump — both halves hit
        pts = sobol_scenarios([0], [1], 16, seed=0).ravel()
        assert (pts < 0.5).sum() >= 6 and (pts >= 0.5).sum() >= 6

    def test_bad_bounds_raise(self):
        with pytest.raises(ValueError):
            sobol_scenarios([0, 0], [1], 4)
        with pytest.raises(ValueError):
            sobol_scenarios([1], [1], 4)


class TestSweep:
    def test_all_pass_when_objective_above_threshold(self):
        res = sweep(lambda p: 5.0, [0], [1], n_scenarios=8, threshold=0.0)
        assert isinstance(res, CoverageResult)
        assert res.n_failures == 0
        assert res.failure_fraction == 0.0

    def test_failure_region_detected(self):
        # value < 0 for x < 0.5 -> roughly half the sweep fails
        res = sweep(lambda p: p[0] - 0.5, [0], [1], n_scenarios=32, threshold=0.0)
        assert 0 < res.n_failures < 32
        assert 0.0 < res.failure_fraction < 1.0

    def test_worst_scenario_reported(self):
        res = sweep(lambda p: (p[0] - 0.3) ** 2, [0], [1], n_scenarios=64, threshold=-1)
        assert res.worst_params[0] == pytest.approx(0.3, abs=0.1)
        assert res.min_value >= 0

    def test_outcomes_cover_every_scenario(self):
        res = sweep(lambda p: 1.0, [0, 0], [1, 1], n_scenarios=16)
        assert len(res.outcomes) == 16
        assert all(o.passed for o in res.outcomes)

    def test_summary_and_dict(self):
        res = sweep(lambda p: -1.0, [0], [1], n_scenarios=8)
        assert "failure region" in res.summary()
        d = res.to_dict()
        assert d["n_failures"] == 8 and d["failure_fraction"] == 1.0
