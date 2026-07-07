"""Unit tests for Clopper-Pearson intervals against known values.

Reference 95% two-sided Clopper-Pearson intervals (from standard tables /
R's binom.test):
  k=0,  n=10 -> [0.0000, 0.3085]
  k=10, n=10 -> [0.6915, 1.0000]   (mirror of k=0)
  k=5,  n=10 -> [0.1871, 0.8129]
  k=3,  n=10 -> [0.0667, 0.6525]
  k=2,  n=20 -> [0.0123, 0.3170]
"""

import pytest

from cotter.stats import clopper_pearson


class TestKnownValues:
    @pytest.mark.parametrize(
        ("k", "n", "lo", "hi"),
        [
            (0, 10, 0.0000, 0.3085),
            (10, 10, 0.6915, 1.0000),
            (5, 10, 0.1871, 0.8129),
            (3, 10, 0.0667, 0.6525),
            (2, 20, 0.0123, 0.3170),
        ],
    )
    def test_intervals_match_tables(self, k, n, lo, hi):
        lower, upper = clopper_pearson(k, n, 0.95)
        assert lower == pytest.approx(lo, abs=1e-4)
        assert upper == pytest.approx(hi, abs=1e-4)


class TestProperties:
    def test_zero_successes_lower_is_zero(self):
        assert clopper_pearson(0, 25)[0] == 0.0

    def test_all_successes_upper_is_one(self):
        assert clopper_pearson(25, 25)[1] == 1.0

    def test_point_estimate_inside_interval(self):
        lower, upper = clopper_pearson(7, 20, 0.95)
        assert lower < 7 / 20 < upper

    def test_higher_confidence_widens_interval(self):
        lo95, hi95 = clopper_pearson(5, 20, 0.95)
        lo99, hi99 = clopper_pearson(5, 20, 0.99)
        assert lo99 < lo95
        assert hi99 > hi95

    def test_symmetry_of_mirrored_counts(self):
        lo_a, hi_a = clopper_pearson(3, 10)
        lo_b, hi_b = clopper_pearson(7, 10)
        assert lo_a == pytest.approx(1 - hi_b, abs=1e-12)
        assert hi_a == pytest.approx(1 - lo_b, abs=1e-12)


class TestValidation:
    @pytest.mark.parametrize("args", [(0, 0), (5, 3), (-1, 10)])
    def test_bad_counts(self, args):
        with pytest.raises(ValueError):
            clopper_pearson(*args)

    @pytest.mark.parametrize("conf", [0.0, 1.0, -0.1, 1.5])
    def test_bad_confidence(self, conf):
        with pytest.raises(ValueError):
            clopper_pearson(5, 10, conf)
