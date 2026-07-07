"""Wald's Sequential Probability Ratio Test for policy success rates.

Tests H0: success rate <= p0 against H1: success rate >= p1, with error
tolerances alpha (probability of accepting H1 when H0 is true) and beta
(probability of accepting H0 when H1 is true). Trials are consumed one at
a time and the test stops as soon as the evidence is decisive, which is
what makes it suitable for expensive simulation rollouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable


class SPRTDecision(str, Enum):
    PASS = "PASS"  # accepted H1: success rate >= p1
    FAIL = "FAIL"  # accepted H0: success rate <= p0
    INCONCLUSIVE = "INCONCLUSIVE"  # hit n_max without crossing a boundary


@dataclass
class SPRTResult:
    decision: SPRTDecision
    n_trials: int
    n_successes: int
    llr: float
    upper_boundary: float
    lower_boundary: float
    p0: float
    p1: float
    alpha: float
    beta: float
    llr_history: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.n_successes / self.n_trials if self.n_trials else float("nan")

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "n_trials": self.n_trials,
            "n_successes": self.n_successes,
            "success_rate": self.success_rate,
            "llr": self.llr,
            "upper_boundary": self.upper_boundary,
            "lower_boundary": self.lower_boundary,
            "p0": self.p0,
            "p1": self.p1,
            "alpha": self.alpha,
            "beta": self.beta,
        }


class SPRT:
    """Incremental SPRT accumulator.

    Feed trial outcomes via :meth:`update`; it returns a decision once the
    log-likelihood ratio crosses a boundary, else ``None``. Use
    :func:`run_sprt` to drive it from a trial-generating callable.
    """

    def __init__(
        self,
        p0: float,
        p1: float,
        alpha: float = 0.05,
        beta: float = 0.05,
        n_max: int = 200,
    ) -> None:
        if not (0.0 < p0 < 1.0 and 0.0 < p1 < 1.0):
            raise ValueError(f"p0 and p1 must be in (0, 1); got p0={p0}, p1={p1}")
        if p1 <= p0:
            raise ValueError(f"require p1 > p0 (H1 must beat H0); got p0={p0}, p1={p1}")
        if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
            raise ValueError(f"alpha and beta must be in (0, 1); got {alpha}, {beta}")
        if n_max < 1:
            raise ValueError(f"n_max must be >= 1; got {n_max}")

        self.p0, self.p1, self.alpha, self.beta, self.n_max = p0, p1, alpha, beta, n_max
        self.upper_boundary = math.log((1.0 - beta) / alpha)
        self.lower_boundary = math.log(beta / (1.0 - alpha))
        self._llr_success = math.log(p1 / p0)
        self._llr_failure = math.log((1.0 - p1) / (1.0 - p0))

        self.llr = 0.0
        self.n_trials = 0
        self.n_successes = 0
        self.llr_history: list[float] = []
        self._decision: SPRTDecision | None = None

    def update(self, success: bool) -> SPRTDecision | None:
        """Record one trial outcome; return the decision if now decisive."""
        if self._decision is not None:
            raise RuntimeError("SPRT already reached a decision; create a new instance")
        if self.n_trials >= self.n_max:
            raise RuntimeError(f"SPRT already consumed n_max={self.n_max} trials")

        self.n_trials += 1
        if success:
            self.n_successes += 1
            self.llr += self._llr_success
        else:
            self.llr += self._llr_failure
        self.llr_history.append(self.llr)

        if self.llr >= self.upper_boundary:
            self._decision = SPRTDecision.PASS
        elif self.llr <= self.lower_boundary:
            self._decision = SPRTDecision.FAIL
        return self._decision

    def result(self) -> SPRTResult:
        decision = self._decision or SPRTDecision.INCONCLUSIVE
        return SPRTResult(
            decision=decision,
            n_trials=self.n_trials,
            n_successes=self.n_successes,
            llr=self.llr,
            upper_boundary=self.upper_boundary,
            lower_boundary=self.lower_boundary,
            p0=self.p0,
            p1=self.p1,
            alpha=self.alpha,
            beta=self.beta,
            llr_history=list(self.llr_history),
        )


def run_sprt(
    trial_fn: Callable[[int], bool] | Iterable[bool],
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
    n_max: int = 200,
) -> SPRTResult:
    """Run an SPRT to completion.

    ``trial_fn`` is either a callable mapping trial index -> success (each
    call may run a fresh simulation rollout) or a pre-collected iterable of
    outcomes. Stops at the first boundary crossing or after ``n_max`` trials.
    """
    sprt = SPRT(p0=p0, p1=p1, alpha=alpha, beta=beta, n_max=n_max)
    if callable(trial_fn):
        outcomes: Iterable[bool] = (trial_fn(i) for i in range(n_max))
    else:
        outcomes = trial_fn
    for outcome in outcomes:
        if sprt.update(bool(outcome)) is not None:
            break
        if sprt.n_trials >= n_max:
            break
    return sprt.result()
