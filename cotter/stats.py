"""Small statistics helpers shared across test categories.

Currently the Clopper-Pearson exact binomial confidence interval, used to
put an interval around every observed success rate (SPRT performance,
adversarial robustness). "Exact" here means the interval is derived from
the binomial CDF (via its Beta-distribution inverse) rather than a normal
approximation, so it stays valid for the small trial counts and rates
near 0 or 1 that this framework routinely produces.
"""

from __future__ import annotations

from scipy import stats


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact confidence interval for a binomial proportion.

    ``k`` successes out of ``n`` trials at the given two-sided
    ``confidence`` level. Returns ``(lower, upper)`` bounds on the true
    success probability. The interval is conservative (coverage >= the
    nominal level). Degenerate counts are handled exactly: ``k == 0`` has
    lower bound 0 and ``k == n`` has upper bound 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if not (0 <= k <= n):
        raise ValueError(f"require 0 <= k <= n; got k={k}, n={n}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")

    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper
