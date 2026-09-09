"""Tests for CMA-ES falsification search (cotter.falsify)."""

import numpy as np
import pytest

pytest.importorskip("cma")

from cotter.falsify import FalsificationResult, falsify


class TestFalsify:
    def test_finds_known_minimum(self):
        # bowl with minimum at (0.7, -0.3), value 0 -> no falsification vs threshold 0
        res = falsify(
            lambda v: (v[0] - 0.7) ** 2 + (v[1] + 0.3) ** 2,
            lower=[-1, -1], upper=[1, 1], max_evaluations=400, seed=1,
        )
        assert isinstance(res, FalsificationResult)
        assert res.best_value == pytest.approx(0.0, abs=1e-2)
        assert res.best_params[0] == pytest.approx(0.7, abs=0.1)
        assert not res.falsified  # min is 0, not < 0

    def test_finds_counterexample_below_threshold(self):
        # objective dips negative near x = -0.8 -> a falsifying region exists
        res = falsify(
            lambda v: (v[0] - 0.5) ** 2 - 1.0,  # min value -1 at x=0.5
            lower=[-1], upper=[1], max_evaluations=200, seed=0,
        )
        assert res.falsified
        assert res.best_value < 0

    def test_stops_early_on_counterexample(self):
        # a scenario immediately below threshold should stop the search fast
        res = falsify(
            lambda v: -5.0,  # always falsifying
            lower=[-1], upper=[1], max_evaluations=500, seed=0,
        )
        assert res.falsified
        assert res.n_evaluations < 50  # terminated well before maxfevals

    def test_history_is_monotone_nonincreasing(self):
        res = falsify(
            lambda v: float(v[0] ** 2), lower=[-2], upper=[2], max_evaluations=100, seed=3,
        )
        assert res.history == sorted(res.history, reverse=True)  # best-so-far only improves

    def test_threshold_controls_falsification(self):
        # a strictly positive bowl; with a positive threshold it "falsifies"
        obj = lambda v: v[0] ** 2 + 0.5
        assert not falsify(obj, [-1], [1], max_evaluations=200, seed=0).falsified
        assert falsify(obj, [-1], [1], max_evaluations=200, seed=0, threshold=1.0).falsified

    def test_mismatched_bounds_raise(self):
        with pytest.raises(ValueError):
            falsify(lambda v: 0.0, lower=[-1, -1], upper=[1])

    def test_degenerate_bounds_raise(self):
        with pytest.raises(ValueError):
            falsify(lambda v: 0.0, lower=[0.0], upper=[0.0])

    def test_summary_and_dict(self):
        res = falsify(lambda v: -1.0, [-1], [1], max_evaluations=50, seed=0)
        assert "FALSIFIED" in res.summary()
        assert res.to_dict()["falsified"] is True
