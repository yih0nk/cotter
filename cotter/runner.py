"""Seeded rollout execution and structured episode records.

The runner drives ``reset()``/``step()`` loops for a black-box policy,
recording per-timestep info dicts (as instrumented by
:class:`cotter.envs.wrapper.CotterWrapper`), per-episode return/length,
and a success flag computed by a caller-supplied predicate. A fixed seed
sequence makes runs reproducible and enables matched-pairs comparisons
between policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import gymnasium as gym
import numpy as np

from cotter.policy import Policy


@dataclass
class EpisodeRecord:
    seed: int
    length: int
    total_reward: float
    terminated: bool
    truncated: bool
    success: bool
    step_infos: list[dict] = field(default_factory=list)

    @property
    def final_info(self) -> dict:
        return self.step_infos[-1] if self.step_infos else {}


# Success predicate: receives (total_reward, length, terminated, truncated,
# final_info) and returns whether the episode counts as a task success.
SuccessFn = Callable[[float, int, bool, bool, dict], bool]


@dataclass
class RolloutSet:
    records: list[EpisodeRecord]

    @property
    def successes(self) -> list[bool]:
        return [r.success for r in self.records]

    @property
    def returns(self) -> list[float]:
        return [r.total_reward for r in self.records]

    @property
    def lengths(self) -> list[int]:
        return [r.length for r in self.records]

    @property
    def seeds(self) -> list[int]:
        return [r.seed for r in self.records]

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes)) if self.records else float("nan")

    @property
    def episode_infos(self) -> list[list[dict]]:
        """Per-trial step-info sequences, the input to evaluate_safety."""
        return [r.step_infos for r in self.records]


def rollout_one(
    policy: Policy,
    env: gym.Env,
    seed: int,
    success_fn: SuccessFn,
    max_steps: int | None = None,
    record_infos: bool = True,
) -> EpisodeRecord:
    """Run a single seeded episode to termination/truncation."""
    obs, info = env.reset(seed=seed)
    step_infos: list[dict] = [info] if record_infos else []
    total_reward, length = 0.0, 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        length += 1
        if record_infos:
            step_infos.append(info)
        if max_steps is not None and length >= max_steps:
            truncated = True

    success = bool(success_fn(total_reward, length, terminated, truncated, info))
    return EpisodeRecord(
        seed=seed,
        length=length,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        success=success,
        step_infos=step_infos,
    )


def make_seed_sequence(n: int, base_seed: int = 0) -> list[int]:
    """Deterministic, well-spread seed list shared across policies."""
    return [int(s) for s in np.random.SeedSequence(base_seed).generate_state(n)]


def run_rollouts(
    policy: Policy,
    env: gym.Env,
    n_episodes: int,
    success_fn: SuccessFn,
    seeds: Sequence[int] | None = None,
    base_seed: int = 0,
    max_steps: int | None = None,
    record_infos: bool = True,
) -> RolloutSet:
    """Run ``n_episodes`` seeded rollouts and return structured records.

    If ``seeds`` is omitted, a deterministic sequence derived from
    ``base_seed`` is used; pass the same seeds to another policy for a
    matched-pairs regression comparison.
    """
    if seeds is None:
        seeds = make_seed_sequence(n_episodes, base_seed)
    elif len(seeds) < n_episodes:
        raise ValueError(f"need {n_episodes} seeds, got {len(seeds)}")

    records = [
        rollout_one(policy, env, int(seeds[i]), success_fn, max_steps, record_infos)
        for i in range(n_episodes)
    ]
    return RolloutSet(records=records)
