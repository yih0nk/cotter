"""Matched-pairs regression testing between two policy versions.

Baseline and candidate are evaluated on the SAME seed sequence, so each
trial is a matched pair sharing identical initial conditions and physics
noise. Pairing means only the *discordant* pairs (where exactly one policy
succeeded) carry information, and McNemar's exact test on those recovers
far more statistical power than an unpaired comparison of aggregate rates.

For continuous per-trial metrics (e.g. episode return) a Wilcoxon
signed-rank variant is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from scipy import stats


class RegressionDecision(str, Enum):
    REGRESSION = "REGRESSION"  # candidate is significantly worse
    NO_REGRESSION = "NO_REGRESSION"


@dataclass
class RegressionResult:
    decision: RegressionDecision
    test: str
    p_value: float
    statistic: float
    alpha: float
    n_pairs: int
    baseline_metric: float
    candidate_metric: float
    detail: dict

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "test": self.test,
            "p_value": self.p_value,
            "statistic": self.statistic,
            "alpha": self.alpha,
            "n_pairs": self.n_pairs,
            "baseline_metric": self.baseline_metric,
            "candidate_metric": self.candidate_metric,
            "detail": self.detail,
        }


def _require_pairs(baseline: Sequence, candidate: Sequence) -> int:
    if len(baseline) != len(candidate):
        raise ValueError(
            f"matched-pairs comparison requires equal trial counts run on the "
            f"same seeds; got baseline={len(baseline)}, candidate={len(candidate)}"
        )
    if len(baseline) == 0:
        raise ValueError("no trials to compare")
    return len(baseline)


def mcnemar_exact(
    baseline_successes: Sequence[bool],
    candidate_successes: Sequence[bool],
    alpha: float = 0.05,
) -> RegressionResult:
    """Exact one-sided McNemar test for 'candidate is worse than baseline'.

    Let b = pairs where baseline succeeded but candidate failed, and
    c = pairs where candidate succeeded but baseline failed. Under H0
    (no difference), b ~ Binomial(b + c, 1/2). The one-sided p-value is
    P(X >= b), small when the candidate loses far more discordant pairs
    than it wins. REGRESSION is declared when p < alpha.
    """
    n = _require_pairs(baseline_successes, candidate_successes)
    b = sum(1 for x, y in zip(baseline_successes, candidate_successes) if x and not y)
    c = sum(1 for x, y in zip(baseline_successes, candidate_successes) if not x and y)

    if b + c == 0:
        p_value = 1.0  # no discordant pairs: no evidence of any difference
    else:
        p_value = stats.binomtest(b, n=b + c, p=0.5, alternative="greater").pvalue

    is_regression = p_value < alpha
    return RegressionResult(
        decision=RegressionDecision.REGRESSION if is_regression else RegressionDecision.NO_REGRESSION,
        test="mcnemar_exact_one_sided",
        p_value=float(p_value),
        statistic=float(b),
        alpha=alpha,
        n_pairs=n,
        baseline_metric=sum(map(bool, baseline_successes)) / n,
        candidate_metric=sum(map(bool, candidate_successes)) / n,
        detail={"discordant_baseline_only": b, "discordant_candidate_only": c},
    )


def wilcoxon_regression(
    baseline_metric: Sequence[float],
    candidate_metric: Sequence[float],
    alpha: float = 0.05,
) -> RegressionResult:
    """One-sided Wilcoxon signed-rank test for 'candidate metric is lower'.

    For continuous per-trial metrics (episode return, completion time as a
    negated value, ...) on the same seed sequence. REGRESSION when the
    candidate's paired metric is significantly lower than the baseline's.
    """
    n = _require_pairs(baseline_metric, candidate_metric)
    diffs = [c - b for b, c in zip(baseline_metric, candidate_metric)]
    if all(d == 0 for d in diffs):
        statistic, p_value = 0.0, 1.0
    else:
        # H1: candidate - baseline shifted below zero
        statistic, p_value = stats.wilcoxon(diffs, alternative="less")

    is_regression = p_value < alpha
    return RegressionResult(
        decision=RegressionDecision.REGRESSION if is_regression else RegressionDecision.NO_REGRESSION,
        test="wilcoxon_signed_rank_one_sided",
        p_value=float(p_value),
        statistic=float(statistic),
        alpha=alpha,
        n_pairs=n,
        baseline_metric=float(sum(baseline_metric) / n),
        candidate_metric=float(sum(candidate_metric) / n),
        detail={"mean_paired_difference": float(sum(diffs) / n)},
    )
