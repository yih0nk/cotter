"""Signal Temporal Logic (STL) specification testing.

STL lets a user state a behavioral requirement — "always speed <= 1.5",
"eventually dist_to_goal <= 0.05", "always (contact -> eventually[0,10]
released)" — and get a *quantitative robustness margin* per rollout: a
signed number saying how safely the spec was satisfied (positive) or how
badly it was violated (negative), not just pass/fail. The margin is the
scoring substrate coverage and falsification build on.

Signals are extracted from the per-step ``info`` dict (see
:func:`extract_dataset`), one named variable per STL identifier. Evaluated
offline with ``rtamt`` (pure-Python discrete-time monitor); imported
lazily so the extra is only needed when STL is actually used
(``pip install cotterbot[stl]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


def _signal_value(info: Mapping, key: str, index: int | None) -> float:
    if key not in info:
        raise KeyError(
            f"STL variable references info key '{key}' but the step info only "
            f"contains {sorted(info.keys())}."
        )
    value = info[key]
    if index is not None:
        return float(np.asarray(value, dtype=float).ravel()[index])
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(
            f"STL variable for info key '{key}' is non-scalar (shape {arr.shape}); "
            "give an 'index' to select a component."
        )
    return float(arr.ravel()[0])


def extract_dataset(steps: Sequence[Mapping], variables: Mapping[str, dict]) -> dict:
    """Build an rtamt dataset ``{'time': [...], var: [...], ...}`` for one episode.

    ``variables`` maps each STL identifier to ``{"key": <info key>,
    "index": <optional int>}``.
    """
    dataset: dict[str, list] = {"time": list(range(len(steps)))}
    for name, spec in variables.items():
        key = spec["key"]
        index = spec.get("index")
        dataset[name] = [_signal_value(info, key, index) for info in steps]
    return dataset


def _make_spec(formula: str, variables: Mapping[str, dict]):
    import rtamt

    spec = rtamt.StlDiscreteTimeSpecification()
    for name in variables:
        spec.declare_var(name, "float")
    spec.spec = formula
    try:
        spec.parse()
    except Exception as exc:  # rtamt raises various parse errors
        raise ValueError(f"could not parse STL formula {formula!r}: {exc}") from exc
    return spec


def _evaluate(spec, dataset: dict) -> float:
    """Run the offline monitor and return the robustness at time 0.

    rtamt's discrete-time offline interpreter needs at least two samples;
    a single-step episode is padded by repeating its one sample so a
    constant trace still yields a well-defined robustness.
    """
    if len(dataset["time"]) < 2:
        dataset = {k: (v + v[-1:] if k != "time" else [0, 1]) for k, v in dataset.items()}
    return float(spec.evaluate(dataset)[0][1])


@dataclass
class STLResult:
    name: str
    formula: str
    threshold: float
    n_episodes: int
    min_robustness: float
    mean_robustness: float
    per_episode: list[float] = field(default_factory=list)
    worst_episode: int = -1
    passed: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "formula": self.formula,
            "threshold": self.threshold,
            "n_episodes": self.n_episodes,
            "min_robustness": self.min_robustness,
            "mean_robustness": self.mean_robustness,
            "worst_episode": self.worst_episode,
            "per_episode": self.per_episode,
            "passed": self.passed,
        }

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}: STL '{self.name}' [{self.formula}] min robustness "
            f"{self.min_robustness:.4g} (mean {self.mean_robustness:.4g}) over "
            f"{self.n_episodes} episodes, threshold {self.threshold:g} "
            f"(worst episode {self.worst_episode})"
        )


def episode_robustness(steps: Sequence[Mapping], formula: str, variables: Mapping[str, dict]) -> float:
    """Robustness of ``formula`` over one episode (the value at time 0)."""
    spec = _make_spec(formula, variables)
    return _evaluate(spec, extract_dataset(steps, variables))


def evaluate_stl(
    episode_infos: Sequence[Sequence[Mapping]],
    formula: str,
    variables: Mapping[str, dict],
    name: str = "stl",
    threshold: float = 0.0,
) -> STLResult:
    """Evaluate an STL spec over rollouts.

    Computes each episode's robustness margin; the spec PASSES iff the
    *worst* (minimum) episode robustness is at or above ``threshold``
    (default 0 — the standard STL satisfaction boundary).
    """
    if not episode_infos:
        raise ValueError("evaluate_stl called with no episodes")
    if not variables:
        raise ValueError("evaluate_stl requires at least one variable")

    spec = _make_spec(formula, variables)  # parse once, fail fast on a bad formula
    per_episode: list[float] = []
    for steps in episode_infos:
        per_episode.append(_evaluate(spec, extract_dataset(steps, variables)))

    min_rob = min(per_episode)
    worst = int(np.argmin(per_episode))
    return STLResult(
        name=name,
        formula=formula,
        threshold=threshold,
        n_episodes=len(per_episode),
        min_robustness=min_rob,
        mean_robustness=float(np.mean(per_episode)),
        per_episode=per_episode,
        worst_episode=worst,
        passed=min_rob >= threshold,
    )
