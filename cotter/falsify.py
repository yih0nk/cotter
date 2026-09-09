"""Falsification: actively search for a scenario that breaks a policy.

Where the test categories *sample* scenarios (fixed seeds, random noise),
falsification *optimizes* over a scenario space to find the worst one —
the initial condition, disturbance, or parameter vector that drives an
objective as low as possible. Point it at an STL-robustness objective
(see :func:`stl_scenario_objective`) and it hunts the scenario that most
violates a spec, returning the concrete counterexample.

Black-box, derivative-free (CMA-ES via ``cma``) so it needs no gradients
and runs on CPU; each evaluation is one simulation rollout. ``cma`` is
imported lazily (the ``[falsify]`` extra).

This is the S-TaLiRo / Breach paradigm: minimize a quantitative
requirement robustness; a negative minimum is a falsifying counterexample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym

    from cotter.policy import Policy


@dataclass
class FalsificationResult:
    best_params: list[float]
    best_value: float
    falsified: bool  # True iff best_value < threshold (a counterexample was found)
    threshold: float
    n_evaluations: int
    history: list[float] = field(default_factory=list)  # best-so-far value per evaluation

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "falsified": self.falsified,
            "threshold": self.threshold,
            "n_evaluations": self.n_evaluations,
        }

    def summary(self) -> str:
        verdict = "FALSIFIED" if self.falsified else "no counterexample"
        return (
            f"{verdict}: min objective {self.best_value:.4g} "
            f"(threshold {self.threshold:g}) after {self.n_evaluations} evaluations"
            + (f"; counterexample at {np.round(self.best_params, 4).tolist()}" if self.falsified else "")
        )


def falsify(
    objective: Callable[[np.ndarray], float],
    lower: Sequence[float],
    upper: Sequence[float],
    max_evaluations: int = 200,
    threshold: float = 0.0,
    sigma0: float = 0.25,
    seed: int = 0,
    x0: Sequence[float] | None = None,
) -> FalsificationResult:
    """Minimize ``objective`` over the box [lower, upper] with CMA-ES.

    Falsification succeeds as soon as (and reports whether) the best value
    found is below ``threshold`` — for an STL-robustness objective,
    ``threshold=0`` means "a spec violation was found". Search stops early
    once a value below ``threshold`` is seen, or at ``max_evaluations``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must have the same shape")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be strictly below its upper bound")

    start = np.asarray(x0, dtype=float) if x0 is not None else (lower + upper) / 2.0

    best_value = float("inf")
    best_params = start.tolist()
    history: list[float] = []
    n_eval = 0
    stop = {"hit": False}

    def wrapped(params) -> float:
        nonlocal best_value, best_params, n_eval
        value = float(objective(np.asarray(params, dtype=float)))
        n_eval += 1
        if value < best_value:
            best_value = value
            best_params = list(map(float, params))
        history.append(best_value)
        if best_value < threshold:
            stop["hit"] = True
        return value

    if lower.size == 1:
        # CMA-ES is unreliable in one dimension; a deterministic sweep of
        # the interval is both robust and a better fit for a 1-D space.
        for x in np.linspace(lower[0], upper[0], max(max_evaluations, 2)):
            wrapped([x])
            if stop["hit"]:
                break
    else:
        import cma

        cma.fmin(
            wrapped,
            list(start),
            sigma0,
            {
                "bounds": [list(lower), list(upper)],
                "maxfevals": max_evaluations,
                "seed": seed,
                "verbose": -9,
                "termination_callback": lambda es: stop["hit"],
            },
        )

    return FalsificationResult(
        best_params=best_params,
        best_value=best_value,
        falsified=best_value < threshold,
        threshold=threshold,
        n_evaluations=n_eval,
        history=history,
    )


def stl_scenario_objective(
    policy: "Policy",
    env_factory: Callable[[], "gym.Env"],
    apply_scenario: Callable[["gym.Env", np.ndarray], None],
    formula: str,
    variables: Mapping[str, dict],
    n_episodes: int = 1,
    base_seed: int = 0,
) -> Callable[[np.ndarray], float]:
    """Build a falsification objective = the STL robustness under a scenario.

    Each call: a fresh env from ``env_factory`` is configured by
    ``apply_scenario(env, params)`` (which should set env parameters that
    persist across reset — masses, frictions, disturbance magnitudes),
    the policy is rolled out, and the spec's worst-episode robustness is
    returned. Minimizing it drives the search toward a spec violation
    (robustness < 0). Pair with :func:`falsify`.
    """
    from cotter.runner import make_seed_sequence, run_rollouts
    from cotter.tests.stl import evaluate_stl

    seeds = make_seed_sequence(n_episodes, base_seed)

    def objective(params: np.ndarray) -> float:
        env = env_factory()
        try:
            apply_scenario(env, params)
            rollouts = run_rollouts(
                policy, env, n_episodes, lambda *a: False, seeds=seeds, record_infos=True
            )
        finally:
            env.close()
        return evaluate_stl(rollouts.episode_infos, formula, variables).min_robustness

    return objective
