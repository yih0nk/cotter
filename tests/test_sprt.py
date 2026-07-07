"""Unit tests for Wald's SPRT against hand-computed values.

Reference configuration used throughout: p0=0.5, p1=0.9, alpha=beta=0.05.
  upper boundary A = ln((1-0.05)/0.05) = ln(19)  ≈  2.9444389792
  lower boundary B = ln(0.05/(1-0.05)) = ln(1/19) ≈ -2.9444389792
  per-success increment  ln(0.9/0.5) = ln(1.8) ≈  0.5877866649
  per-failure increment  ln(0.1/0.5) = ln(0.2) ≈ -1.6094379124
"""

import math

import pytest

from cotter.stats import clopper_pearson
from cotter.tests.sprt import SPRT, SPRTDecision, run_sprt

A = math.log(19)
SUCC = math.log(1.8)
FAIL = math.log(0.2)


def make(n_max=200):
    return SPRT(p0=0.5, p1=0.9, alpha=0.05, beta=0.05, n_max=n_max)


class TestBoundaries:
    def test_boundary_values(self):
        s = make()
        assert s.upper_boundary == pytest.approx(2.9444389792, abs=1e-9)
        assert s.lower_boundary == pytest.approx(-2.9444389792, abs=1e-9)

    def test_asymmetric_errors(self):
        s = SPRT(p0=0.5, p1=0.9, alpha=0.01, beta=0.10)
        # A = ln(0.90/0.01) = ln(90), B = ln(0.10/0.99)
        assert s.upper_boundary == pytest.approx(math.log(90.0), abs=1e-12)
        assert s.lower_boundary == pytest.approx(math.log(0.10 / 0.99), abs=1e-12)


class TestHandComputedSequences:
    def test_all_successes_passes_at_trial_six(self):
        # n * ln(1.8) first exceeds ln(19) at n = 6:
        #   5 * 0.5877866649 = 2.9389333 < 2.9444390
        #   6 * 0.5877866649 = 3.5267200 >= 2.9444390
        s = make()
        for i in range(5):
            assert s.update(True) is None, f"crossed too early at trial {i + 1}"
        assert s.update(True) == SPRTDecision.PASS
        res = s.result()
        assert res.n_trials == 6
        assert res.n_successes == 6
        assert res.llr == pytest.approx(6 * SUCC, abs=1e-9)
        assert res.success_rate == 1.0

    def test_all_failures_fails_at_trial_two(self):
        # n * ln(0.2) first drops below -ln(19) at n = 2:
        #   1 * -1.6094379 = -1.6094379 > -2.9444390
        #   2 * -1.6094379 = -3.2188758 <= -2.9444390
        s = make()
        assert s.update(False) is None
        assert s.update(False) == SPRTDecision.FAIL
        res = s.result()
        assert res.n_trials == 2
        assert res.n_successes == 0
        assert res.llr == pytest.approx(2 * FAIL, abs=1e-9)

    def test_mixed_sequence_recovers_and_passes(self):
        # S F S S S S S S S: llr = k*ln(1.8) + m*ln(0.2); crosses A at trial 9
        # with 8 successes, 1 failure: 8*0.5877867 - 1.6094379 = 3.0928554 >= A
        outcomes = [True, False] + [True] * 7
        res = run_sprt(iter(outcomes), p0=0.5, p1=0.9, alpha=0.05, beta=0.05)
        assert res.decision == SPRTDecision.PASS
        assert res.n_trials == 9
        assert res.llr == pytest.approx(8 * SUCC + FAIL, abs=1e-9)

    def test_early_stop_ignores_remaining_trials(self):
        calls = []

        def trial(i):
            calls.append(i)
            return False

        res = run_sprt(trial, p0=0.5, p1=0.9, alpha=0.05, beta=0.05, n_max=100)
        assert res.decision == SPRTDecision.FAIL
        assert res.n_trials == 2
        assert calls == [0, 1]  # stopped sampling immediately after crossing


class TestNMaxCutoff:
    def test_alternating_outcomes_inconclusive(self):
        # p0=0.4, p1=0.6: increments are +/- ln(1.5), so S,F,S,F,... oscillates
        # between ln(1.5) and 0 and never approaches +/- ln(19).
        outcomes = [i % 2 == 0 for i in range(10)]
        res = run_sprt(iter(outcomes), p0=0.4, p1=0.6, alpha=0.05, beta=0.05, n_max=10)
        assert res.decision == SPRTDecision.INCONCLUSIVE
        assert res.n_trials == 10
        assert res.llr == pytest.approx(0.0, abs=1e-9)
        assert res.success_rate == pytest.approx(0.5)


class TestValidationAndStateGuards:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"p0": 0.9, "p1": 0.5},  # p1 <= p0
            {"p0": 0.5, "p1": 0.5},
            {"p0": 0.0, "p1": 0.9},  # boundary probabilities
            {"p0": 0.5, "p1": 1.0},
            {"p0": 0.5, "p1": 0.9, "alpha": 0.0},
            {"p0": 0.5, "p1": 0.9, "beta": 1.0},
            {"p0": 0.5, "p1": 0.9, "n_max": 0},
        ],
    )
    def test_invalid_parameters_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SPRT(**{"alpha": 0.05, "beta": 0.05, **kwargs})

    def test_update_after_decision_raises(self):
        s = make()
        s.update(False)
        s.update(False)  # decision reached
        with pytest.raises(RuntimeError):
            s.update(True)

    def test_result_dict_roundtrip(self):
        res = run_sprt(iter([True] * 6), p0=0.5, p1=0.9)
        d = res.to_dict()
        assert d["decision"] == "PASS"
        assert d["n_trials"] == 6
        assert d["success_rate"] == 1.0
        assert d["ci_level"] == 0.95
        assert d["ci_lower"] == pytest.approx(clopper_pearson(6, 6)[0])
        assert d["ci_upper"] == 1.0


class TestConfidenceInterval:
    def test_ci_matches_clopper_pearson(self):
        # 8 successes / 1 failure over 9 trials in a mixed pass sequence
        res = run_sprt(iter([True, False] + [True] * 7), p0=0.5, p1=0.9)
        assert res.n_trials == 9 and res.n_successes == 8
        lo, hi = clopper_pearson(8, 9, 0.95)
        assert res.ci_lower == pytest.approx(lo)
        assert res.ci_upper == pytest.approx(hi)

    def test_point_estimate_within_ci(self):
        res = run_sprt(iter([True, False] + [True] * 7), p0=0.5, p1=0.9)
        assert res.ci_lower <= res.success_rate <= res.ci_upper
