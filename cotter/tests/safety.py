"""Per-timestep hard-limit safety checks.

A safety test is a set of :class:`SafetyLimit` objects, each naming a
quantity that must appear in the per-step ``info`` dict (scalar or array)
and a maximum allowed absolute value. Every timestep of every trial is
checked; a single violation anywhere fails the test. There is no
averaging — this mirrors how hard physical limits (ISO-style force and
velocity caps) are specified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class SafetyDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class SafetyLimit:
    """|value| of ``quantity`` must stay <= ``max_abs`` at every timestep."""

    quantity: str
    max_abs: float

    def __post_init__(self) -> None:
        if self.max_abs <= 0:
            raise ValueError(f"max_abs must be positive; got {self.max_abs}")


@dataclass(frozen=True)
class SafetyViolation:
    trial: int
    timestep: int
    quantity: str
    value: float
    limit: float
    element: int | None = None  # index within a vector quantity, None for scalars

    def to_dict(self) -> dict:
        return {
            "trial": self.trial,
            "timestep": self.timestep,
            "quantity": self.quantity,
            "value": self.value,
            "limit": self.limit,
            "element": self.element,
        }


@dataclass
class SafetyResult:
    decision: SafetyDecision
    limits: list[SafetyLimit]
    n_trials: int
    n_timesteps_checked: int
    violations: list[SafetyViolation] = field(default_factory=list)
    worst_observed: dict[str, float] = field(default_factory=dict)

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "limits": [{"quantity": l.quantity, "max_abs": l.max_abs} for l in self.limits],
            "n_trials": self.n_trials,
            "n_timesteps_checked": self.n_timesteps_checked,
            "n_violations": self.n_violations,
            "violations": [v.to_dict() for v in self.violations[:50]],
            "worst_observed": self.worst_observed,
        }


def check_step(
    info: Mapping,
    limits: Sequence[SafetyLimit],
    trial: int = 0,
    timestep: int = 0,
) -> list[SafetyViolation]:
    """Check one timestep's info dict against all limits.

    Raises ``KeyError`` if a limited quantity is missing from ``info`` —
    a limit on a quantity the environment does not expose is a
    configuration bug and must not silently pass.
    """
    violations: list[SafetyViolation] = []
    for limit in limits:
        if limit.quantity not in info:
            raise KeyError(
                f"safety limit references quantity '{limit.quantity}' but the "
                f"step info dict only contains {sorted(info.keys())}. Wrap the "
                "environment so this quantity is exposed (see cotter.envs.wrapper)."
            )
        value = np.atleast_1d(np.asarray(info[limit.quantity], dtype=float))
        is_scalar = value.size == 1 and np.ndim(info[limit.quantity]) == 0
        for idx, v in enumerate(value.ravel()):
            if abs(v) > limit.max_abs:
                violations.append(
                    SafetyViolation(
                        trial=trial,
                        timestep=timestep,
                        quantity=limit.quantity,
                        value=float(v),
                        limit=limit.max_abs,
                        element=None if is_scalar else idx,
                    )
                )
    return violations


def evaluate_safety(
    episode_infos: Sequence[Sequence[Mapping]],
    limits: Sequence[SafetyLimit],
) -> SafetyResult:
    """Evaluate hard limits over recorded rollouts.

    ``episode_infos`` is one sequence of per-step info dicts per trial
    (as recorded by :mod:`cotter.runner`). ANY single violation across
    ANY timestep of ANY trial fails the test.
    """
    if not limits:
        raise ValueError("evaluate_safety called with no limits configured")

    violations: list[SafetyViolation] = []
    worst: dict[str, float] = {limit.quantity: 0.0 for limit in limits}
    n_steps = 0

    for trial, steps in enumerate(episode_infos):
        for timestep, info in enumerate(steps):
            n_steps += 1
            violations.extend(check_step(info, limits, trial=trial, timestep=timestep))
            for limit in limits:
                values = np.abs(np.asarray(info[limit.quantity], dtype=float))
                if values.size == 0:
                    continue  # e.g. a mocap-controlled env has no actuator forces
                observed = float(np.max(values))
                if observed > worst[limit.quantity]:
                    worst[limit.quantity] = observed

    return SafetyResult(
        decision=SafetyDecision.FAIL if violations else SafetyDecision.PASS,
        limits=list(limits),
        n_trials=len(episode_infos),
        n_timesteps_checked=n_steps,
        violations=violations,
        worst_observed=worst,
    )
