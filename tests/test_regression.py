"""Unit tests for matched-pairs regression statistics.

McNemar hand computations (one-sided exact, H1: candidate worse):
  b = baseline-only successes, c = candidate-only successes,
  p = P(X >= b) with X ~ Binomial(b + c, 1/2).

  b=8, c=2:  p = (C(10,8)+C(10,9)+C(10,10)) / 2^10 = (45+10+1)/1024 = 0.0546875
  b=9, c=1:  p = (10+1)/1024                                        = 0.0107421875
  b=5, c=0:  p = 1/32                                               = 0.03125
  b=1, c=9:  p = 1 - C(10,0)/2^10 = 1023/1024                       = 0.9990234375
"""

import pytest

from cotter.tests.regression import (
    RegressionDecision,
    mcnemar_exact,
    wilcoxon_regression,
)


def paired_outcomes(b, c, both_succeed=10, both_fail=5):
    """Build matched-pair outcome vectors with the given discordant counts."""
    baseline, candidate = [], []
    baseline += [True] * b;  candidate += [False] * b
    baseline += [False] * c; candidate += [True] * c
    baseline += [True] * both_succeed;  candidate += [True] * both_succeed
    baseline += [False] * both_fail;    candidate += [False] * both_fail
    return baseline, candidate


class TestMcNemarHandComputed:
    def test_b8_c2_just_misses_significance(self):
        res = mcnemar_exact(*paired_outcomes(b=8, c=2), alpha=0.05)
        assert res.p_value == pytest.approx(56 / 1024, abs=1e-12)
        assert res.decision == RegressionDecision.NO_REGRESSION
        assert res.statistic == 8
        assert res.detail == {"discordant_baseline_only": 8, "discordant_candidate_only": 2}

    def test_b9_c1_is_regression(self):
        res = mcnemar_exact(*paired_outcomes(b=9, c=1), alpha=0.05)
        assert res.p_value == pytest.approx(11 / 1024, abs=1e-12)
        assert res.decision == RegressionDecision.REGRESSION

    def test_b5_c0_is_regression(self):
        res = mcnemar_exact(*paired_outcomes(b=5, c=0), alpha=0.05)
        assert res.p_value == pytest.approx(1 / 32, abs=1e-12)
        assert res.decision == RegressionDecision.REGRESSION

    def test_candidate_better_never_regression(self):
        res = mcnemar_exact(*paired_outcomes(b=1, c=9), alpha=0.05)
        assert res.p_value == pytest.approx(1023 / 1024, abs=1e-12)
        assert res.decision == RegressionDecision.NO_REGRESSION

    def test_identical_outcomes_p_one(self):
        res = mcnemar_exact(*paired_outcomes(b=0, c=0), alpha=0.05)
        assert res.p_value == 1.0
        assert res.decision == RegressionDecision.NO_REGRESSION

    def test_concordant_pairs_do_not_affect_p(self):
        a = mcnemar_exact(*paired_outcomes(b=9, c=1, both_succeed=0, both_fail=0))
        b = mcnemar_exact(*paired_outcomes(b=9, c=1, both_succeed=50, both_fail=50))
        assert a.p_value == pytest.approx(b.p_value, abs=1e-15)

    def test_success_rates_reported(self):
        res = mcnemar_exact(*paired_outcomes(b=8, c=2, both_succeed=10, both_fail=5))
        assert res.n_pairs == 25
        assert res.baseline_metric == pytest.approx(18 / 25)
        assert res.candidate_metric == pytest.approx(12 / 25)


class TestValidation:
    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="same seeds"):
            mcnemar_exact([True, False], [True])

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            mcnemar_exact([], [])


class TestWilcoxon:
    def test_consistently_lower_candidate_is_regression(self):
        baseline = [100.0, 98.0, 102.0, 99.0, 101.0, 100.5, 97.0, 103.0]
        candidate = [x - 5.0 for x in baseline]
        res = wilcoxon_regression(baseline, candidate, alpha=0.05)
        assert res.decision == RegressionDecision.REGRESSION
        assert res.p_value < 0.05
        assert res.detail["mean_paired_difference"] == pytest.approx(-5.0)

    def test_identical_metrics_no_regression(self):
        vals = [1.0, 2.0, 3.0]
        res = wilcoxon_regression(vals, list(vals))
        assert res.p_value == 1.0
        assert res.decision == RegressionDecision.NO_REGRESSION

    def test_improved_candidate_no_regression(self):
        baseline = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2]
        candidate = [x + 3.0 for x in baseline]
        res = wilcoxon_regression(baseline, candidate)
        assert res.decision == RegressionDecision.NO_REGRESSION
        assert res.p_value > 0.5


class TestEffectSizes:
    def test_mcnemar_odds_ratio(self):
        res = mcnemar_exact(*paired_outcomes(b=9, c=1))
        assert res.effect_size_name == "discordant_odds_ratio"
        assert res.effect_size == pytest.approx(9.0)  # 9/1
        assert res.to_dict()["effect_size"] == pytest.approx(9.0)

    def test_mcnemar_odds_ratio_infinite_when_candidate_never_wins(self):
        res = mcnemar_exact(*paired_outcomes(b=5, c=0))
        assert res.effect_size == float("inf")

    def test_mcnemar_odds_ratio_nan_without_discordant_pairs(self):
        import math

        res = mcnemar_exact(*paired_outcomes(b=0, c=0))
        assert math.isnan(res.effect_size)

    def test_wilcoxon_rank_biserial_all_worse_is_minus_one(self):
        # every paired difference negative -> rank-biserial = -1
        baseline = [100.0, 98.0, 102.0, 99.0, 101.0, 100.5, 97.0, 103.0]
        candidate = [x - 5.0 for x in baseline]
        res = wilcoxon_regression(baseline, candidate)
        assert res.effect_size_name == "rank_biserial_correlation"
        assert res.effect_size == pytest.approx(-1.0)

    def test_wilcoxon_rank_biserial_all_better_is_plus_one(self):
        baseline = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2]
        candidate = [x + 3.0 for x in baseline]
        res = wilcoxon_regression(baseline, candidate)
        assert res.effect_size == pytest.approx(1.0)

    def test_wilcoxon_rank_biserial_zero_when_identical(self):
        vals = [1.0, 2.0, 3.0]
        assert wilcoxon_regression(vals, list(vals)).effect_size == 0.0
