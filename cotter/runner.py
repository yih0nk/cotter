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

DEFAULT_N_WORKERS = 4


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

    def metric_values(self, metric: str = "return") -> list[float]:
        """Per-episode values of a regression metric.

        ``"return"`` gives total episode reward; any other name reads that
        key from each episode's final-step info (e.g. ``"x_position"`` for
        locomotion distance), which requires rollouts recorded with
        ``record_infos=True``.
        """
        if metric == "return":
            return self.returns
        values = []
        for record in self.records:
            final = record.final_info
            if metric not in final:
                raise KeyError(
                    f"regression metric '{metric}' not found in the final step "
                    f"info (keys: {sorted(final.keys())}); rollouts must be run "
                    "with record_infos=True to use an info-based metric"
                )
            values.append(float(final[metric]))
        return values


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


def _extract_env_info(batched: dict, i: int) -> dict:
    """Reconstruct env ``i``'s per-step info dict from a vector env's batched
    info (dict of stacked arrays with ``_key`` presence masks)."""
    out: dict = {}
    for key, values in batched.items():
        if key.startswith("_"):
            continue
        mask = batched.get("_" + key)
        if mask is not None and not bool(mask[i]):
            continue
        value = values[i]
        out[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return out


def run_rollouts_parallel(
    policy: Policy,
    env_factory: Callable[[], gym.Env],
    n_episodes: int,
    success_fn: SuccessFn,
    seeds: Sequence[int] | None = None,
    base_seed: int = 0,
    max_steps: int | None = None,
    record_infos: bool = True,
    n_workers: int = DEFAULT_N_WORKERS,
) -> RolloutSet:
    """Run seeded rollouts across worker processes via ``AsyncVectorEnv``.

    Produces results bit-identical to :func:`run_rollouts` on the same
    seeds: each episode runs in an isolated env seeded with its own seed,
    the policy is evaluated deterministically per observation in the main
    process, and seeds are consumed in disjoint chunks of ``n_workers``.
    Only the environment stepping is parallelized. Falls back to the
    serial runner when ``n_workers <= 1``.
    """
    if seeds is None:
        seeds = make_seed_sequence(n_episodes, base_seed)
    elif len(seeds) < n_episodes:
        raise ValueError(f"need {n_episodes} seeds, got {len(seeds)}")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1; got {n_workers}")
    seeds = [int(s) for s in seeds[:n_episodes]]

    if n_workers == 1:
        return run_rollouts(
            policy, env_factory(), n_episodes, success_fn,
            seeds=seeds, max_steps=max_steps, record_infos=record_infos,
        )

    records: list[EpisodeRecord | None] = [None] * n_episodes
    for start in range(0, n_episodes, n_workers):
        chunk = seeds[start : start + n_workers]
        for offset, record in enumerate(_run_chunk(
            policy, env_factory, chunk, success_fn, max_steps, record_infos
        )):
            records[start + offset] = record
    return RolloutSet(records=[r for r in records if r is not None])


def _run_chunk(
    policy: Policy,
    env_factory: Callable[[], gym.Env],
    chunk_seeds: list[int],
    success_fn: SuccessFn,
    max_steps: int | None,
    record_infos: bool,
) -> list[EpisodeRecord]:
    """Run one chunk of episodes (one per worker) in lockstep, stopping to
    record each env at its own termination while others keep stepping."""
    k = len(chunk_seeds)
    venv = gym.vector.AsyncVectorEnv([env_factory] * k)
    try:
        obs, info = venv.reset(seed=list(chunk_seeds))
        rewards = [0.0] * k
        lengths = [0] * k
        terminated = [False] * k
        truncated = [False] * k
        done = [False] * k
        step_infos: list[list[dict]] = [
            [_extract_env_info(info, i)] if record_infos else [] for i in range(k)
        ]
        final_info = [_extract_env_info(info, i) for i in range(k)]

        while not all(done):
            actions = np.stack([np.asarray(policy.predict(obs[i])) for i in range(k)])
            obs, reward, term, trunc, info = venv.step(actions)
            for i in range(k):
                if done[i]:
                    continue
                rewards[i] += float(reward[i])
                lengths[i] += 1
                env_info = _extract_env_info(info, i)
                final_info[i] = env_info
                if record_infos:
                    step_infos[i].append(env_info)
                hit_cap = max_steps is not None and lengths[i] >= max_steps
                if bool(term[i]) or bool(trunc[i]) or hit_cap:
                    done[i] = True
                    terminated[i] = bool(term[i])
                    truncated[i] = bool(trunc[i]) or hit_cap
    finally:
        venv.close()

    records = []
    for i in range(k):
        success = bool(success_fn(
            rewards[i], lengths[i], terminated[i], truncated[i], final_info[i]
        ))
        records.append(EpisodeRecord(
            seed=chunk_seeds[i],
            length=lengths[i],
            total_reward=rewards[i],
            terminated=terminated[i],
            truncated=truncated[i],
            success=success,
            step_infos=step_infos[i],
        ))
    return records
