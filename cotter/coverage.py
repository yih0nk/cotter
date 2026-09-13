"""Operating-envelope coverage sweeps.

A green report is only trustworthy if it exercised the envelope it claims
to cover. Coverage sampling sweeps the scenario space — initial states,
masses, frictions, disturbance ranges — with a low-discrepancy Sobol
sequence (even coverage, no clustering) and reports the metric (success
rate, STL robustness, …) *as a function of the scenario*, surfacing the
regions that fail rather than a single aggregate number.

Pairs with :mod:`cotter.falsify`: the same ``objective(params) -> float``
(e.g. :func:`cotter.falsify.stl_scenario_objective`) is swept here for
coverage and searched there for the worst case. Pure CPU — each scenario
is one rollout — and Sobol comes from ``scipy.stats.qmc`` (already a
dependency), so no extra install.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


def sobol_scenarios(
    lower: Sequence[float], upper: Sequence[float], n: int, seed: int = 0
) -> np.ndarray:
    """``n`` low-discrepancy scenarios in the box [lower, upper], shape (n, dim)."""
    from scipy.stats import qmc

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must have the same shape")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly below its upper bound")
    if n < 1:
        raise ValueError("n must be >= 1")
    with warnings.catch_warnings():
        # Sobol is best balanced at powers of two; a non-2^m count still
        # samples correctly, just without that guarantee — don't nag.
        warnings.simplefilter("ignore")
        unit = qmc.Sobol(d=lower.size, seed=seed).random(n)
    return qmc.scale(unit, lower, upper)


@dataclass(frozen=True)
class ScenarioOutcome:
    params: list[float]
    value: float
    passed: bool

    def to_dict(self) -> dict:
        return {"params": self.params, "value": self.value, "passed": self.passed}


@dataclass
class CoverageResult:
    n_scenarios: int
    threshold: float
    n_failures: int
    min_value: float
    worst_params: list[float]
    outcomes: list[ScenarioOutcome] = field(default_factory=list)

    @property
    def failure_fraction(self) -> float:
        return self.n_failures / self.n_scenarios if self.n_scenarios else 0.0

    def to_dict(self) -> dict:
        return {
            "n_scenarios": self.n_scenarios,
            "threshold": self.threshold,
            "n_failures": self.n_failures,
            "failure_fraction": self.failure_fraction,
            "min_value": self.min_value,
            "worst_params": self.worst_params,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def summary(self) -> str:
        return (
            f"covered {self.n_scenarios} scenarios, "
            f"{self.n_failures} below threshold {self.threshold:g} "
            f"({self.failure_fraction:.0%} failure region); "
            f"min value {self.min_value:.4g} at {np.round(self.worst_params, 4).tolist()}"
        )


def sweep(
    objective: Callable[[np.ndarray], float],
    lower: Sequence[float],
    upper: Sequence[float],
    n_scenarios: int = 64,
    threshold: float = 0.0,
    seed: int = 0,
) -> CoverageResult:
    """Evaluate ``objective`` over a Sobol sweep of the scenario box.

    A scenario "fails" when its value falls below ``threshold`` (for an STL
    robustness objective, ``threshold=0`` counts spec violations). Returns
    every scenario's value plus the failure fraction and worst scenario —
    a coverage map, not a single number.
    """
    scenarios = sobol_scenarios(lower, upper, n_scenarios, seed)
    outcomes: list[ScenarioOutcome] = []
    min_value = float("inf")
    worst = list(scenarios[0])
    n_fail = 0
    for params in scenarios:
        value = float(objective(np.asarray(params, dtype=float)))
        passed = value >= threshold
        n_fail += not passed
        if value < min_value:
            min_value = value
            worst = list(map(float, params))
        outcomes.append(ScenarioOutcome(list(map(float, params)), value, passed))

    return CoverageResult(
        n_scenarios=len(outcomes),
        threshold=threshold,
        n_failures=n_fail,
        min_value=min_value,
        worst_params=worst,
        outcomes=outcomes,
    )
