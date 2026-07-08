"""Adversarial robustness testing via bounded observation perturbations.

The adversary attacks the victim policy's *perception*: at every timestep
it chooses a perturbation delta with ||delta||_inf <= epsilon which is
added to the observation the victim sees. The environment's true state is
untouched and the victim's weights are never modified — this models
sensor noise, calibration error, and worst-case estimation drift.

Two adversaries are provided:

* :class:`RandomAdversary` — i.i.d. uniform noise in the budget. Always
  works; this is the guaranteed floor for the test category.
* a learned adversary trained with SB3 PPO on
  :class:`ObservationPerturbationEnv`, where the adversary's action is the
  perturbation and its reward is the *negative* of the victim's reward.
  :func:`train_adversary` falls back to the random baseline (with a clear
  note in the result) if training fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from cotter.policy import Policy
from cotter.stats import clopper_pearson
from cotter.runner import (
    SuccessFn,
    make_seed_sequence,
    run_rollouts,
    run_rollouts_parallel,
)


class Adversary(Protocol):
    name: str

    def perturb(self, obs) -> np.ndarray:
        """Given the full observation, return a delta for the perturbable
        (target) array, with ``||delta||_inf <= epsilon``."""
        ...


def _target_array(obs, target_key: str):
    return obs[target_key] if isinstance(obs, dict) else obs


class RandomAdversary:
    """Uniform i.i.d. noise within the L-inf budget. The guaranteed floor."""

    def __init__(self, epsilon: float, seed: int = 0, target_key: str = "observation") -> None:
        self.epsilon = epsilon
        self.name = "random"
        self.target_key = target_key
        self._rng = np.random.default_rng(seed)

    def perturb(self, obs) -> np.ndarray:
        arr = _target_array(obs, self.target_key)
        return self._rng.uniform(-self.epsilon, self.epsilon, size=np.shape(arr))


class NullAdversary:
    """Zero perturbation; useful for sanity checks."""

    def __init__(self, target_key: str = "observation") -> None:
        self.name = "null"
        self.target_key = target_key

    def perturb(self, obs) -> np.ndarray:
        return np.zeros_like(np.asarray(_target_array(obs, self.target_key), dtype=float))


class PPOAdversary:
    """A trained SB3 model driving the perturbation, scaled into the budget.

    The model observes the full (possibly Dict) observation and outputs a
    perturbation for the target array, so ``perturb`` is given the whole
    observation.
    """

    def __init__(self, model, epsilon: float) -> None:
        self.name = "ppo"
        self.epsilon = epsilon
        self.model = model

    def perturb(self, obs) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.epsilon


DEFAULT_TARGET_KEY = "observation"


def perturbable_shape(obs_space: gym.Space, target_key: str = DEFAULT_TARGET_KEY):
    """Shape of the array the adversary perturbs, or raise if unsupported.

    Box spaces are perturbed whole. Dict spaces (gymnasium-robotics style)
    are perturbed only on their ``target_key`` sub-array — the *sensed*
    state — leaving goal keys (the task specification) untouched.
    """
    if isinstance(obs_space, gym.spaces.Box):
        return obs_space.shape
    if isinstance(obs_space, gym.spaces.Dict):
        sub = obs_space.spaces.get(target_key)
        if not isinstance(sub, gym.spaces.Box):
            raise TypeError(
                f"adversarial perturbation of a Dict observation targets the "
                f"'{target_key}' key, which must be a Box; available Box keys: "
                f"{[k for k, v in obs_space.spaces.items() if isinstance(v, gym.spaces.Box)]}"
            )
        return sub.shape
    raise TypeError(
        f"adversarial testing supports Box or Dict observation spaces; got "
        f"{type(obs_space).__name__}"
    )


def apply_perturbation(obs, delta, target_key: str = DEFAULT_TARGET_KEY):
    """Add ``delta`` to a (possibly Dict) observation's perturbable part."""
    if isinstance(obs, dict):
        perturbed = dict(obs)
        perturbed[target_key] = obs[target_key] + delta
        return perturbed
    return obs + delta


class ObservationPerturbationEnv(gym.Env):
    """The attack problem phrased as an RL environment for the adversary.

    Observation: the true environment observation. Action: a vector in
    [-1, 1]^k, scaled by epsilon and added to what the frozen victim sees
    (the whole Box obs, or the ``target_key`` sub-array of a Dict obs).
    Reward: negative of the victim's reward, so maximizing adversary
    return minimizes victim performance.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, env: gym.Env, victim: Policy, epsilon: float,
        target_key: str = DEFAULT_TARGET_KEY,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive; got {epsilon}")
        self.env = env
        self.victim = victim
        self.epsilon = epsilon
        self.target_key = target_key
        self.observation_space = env.observation_space
        pert_shape = perturbable_shape(env.observation_space, target_key)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=pert_shape, dtype=np.float32)
        self._true_obs = None

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._true_obs = obs
        return obs, info

    def step(self, action):
        delta = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.epsilon
        perturbed = apply_perturbation(self._true_obs, delta, self.target_key)
        victim_action = self.victim.predict(perturbed)
        obs, reward, terminated, truncated, info = self.env.step(victim_action)
        self._true_obs = obs
        return obs, -float(reward), terminated, truncated, info

    def close(self):
        self.env.close()


class _PerturbedPolicy:
    """The victim as seen through the adversary: predict on perturbed obs.

    Runs the full attack loop through the ordinary rollout runner so the
    adversarial evaluation records the same instrumented step infos as
    every other test category.
    """

    def __init__(self, victim: Policy, adversary: Adversary,
                 target_key: str = DEFAULT_TARGET_KEY) -> None:
        self.victim = victim
        self.adversary = adversary
        self.target_key = target_key
        self.name = f"{victim.name}+{adversary.name}"

    def predict(self, obs) -> np.ndarray:
        # the adversary sees the full observation and returns a delta for
        # the target array; apply it to what the victim then perceives
        delta = self.adversary.perturb(obs)
        return self.victim.predict(apply_perturbation(obs, delta, self.target_key))


@dataclass
class AdversarialResult:
    adversary_type: str
    epsilon: float
    norm: str
    clean_success_rate: float
    adversarial_success_rate: float
    n_episodes: int
    min_success_rate: float
    passed: bool
    notes: str = ""
    ci_lower: float = float("nan")  # CI on the adversarial (attacked) success rate
    ci_upper: float = float("nan")
    ci_level: float = 0.95

    @property
    def success_rate_drop(self) -> float:
        return self.clean_success_rate - self.adversarial_success_rate

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        text = (
            f"{verdict}: success rate {self.clean_success_rate:.1%} clean -> "
            f"{self.adversarial_success_rate:.1%} "
            f"[{self.ci_lower:.1%}, {self.ci_upper:.1%}] under {self.adversary_type} "
            f"{self.norm} perturbation (eps={self.epsilon}, n={self.n_episodes}, "
            f"required >= {self.min_success_rate:.0%})"
        )
        if self.notes:
            text += f" [{self.notes}]"
        return text

    def to_dict(self) -> dict:
        return {
            "adversary_type": self.adversary_type,
            "epsilon": self.epsilon,
            "norm": self.norm,
            "clean_success_rate": self.clean_success_rate,
            "adversarial_success_rate": self.adversarial_success_rate,
            "adversarial_ci_lower": self.ci_lower,
            "adversarial_ci_upper": self.ci_upper,
            "ci_level": self.ci_level,
            "success_rate_drop": self.success_rate_drop,
            "n_episodes": self.n_episodes,
            "min_success_rate": self.min_success_rate,
            "passed": self.passed,
            "notes": self.notes,
        }


def run_adversarial_test(
    victim: Policy,
    env: gym.Env,
    success_fn: SuccessFn,
    epsilon: float,
    n_episodes: int = 20,
    adversary: Adversary | None = None,
    min_success_rate: float = 0.5,
    base_seed: int = 0,
    seeds: Sequence[int] | None = None,
    notes: str = "",
    n_workers: int = 1,
    env_factory: Callable[[], gym.Env] | None = None,
    ci_level: float = 0.95,
    target_key: str = DEFAULT_TARGET_KEY,
) -> AdversarialResult:
    """Measure worst-case success rate under bounded observation attack.

    Clean and perturbed rollouts share the same seed sequence, so the
    reported drop is a matched comparison. PASS requires the perturbed
    success rate to stay at or above ``min_success_rate``. Pass
    ``n_workers > 1`` with an ``env_factory`` to evaluate the fixed-N
    rollouts in parallel (the adversary and victim stay in the main
    process, so results are identical to the serial path). Dict
    observations are perturbed only on ``target_key`` (the sensed state).
    """
    # Validate the obs space is perturbable (Box, or Dict with a Box
    # target key); raises a clear TypeError otherwise.
    perturbable_shape(env.observation_space, target_key)
    if adversary is None:
        adversary = RandomAdversary(epsilon, seed=base_seed, target_key=target_key)
    if seeds is None:
        seeds = make_seed_sequence(n_episodes, base_seed)

    def rollouts(pol):
        if n_workers > 1:
            if env_factory is None:
                raise ValueError("n_workers > 1 requires an env_factory")
            return run_rollouts_parallel(
                pol, env_factory, n_episodes, success_fn,
                seeds=seeds, record_infos=False, n_workers=n_workers,
            )
        return run_rollouts(pol, env, n_episodes, success_fn, seeds=seeds, record_infos=False)

    clean = rollouts(victim)
    attacked = rollouts(_PerturbedPolicy(victim, adversary, target_key))

    attacked_successes = sum(1 for s in attacked.successes if s)
    ci_lower, ci_upper = clopper_pearson(attacked_successes, n_episodes, ci_level)

    return AdversarialResult(
        adversary_type=adversary.name,
        epsilon=epsilon,
        norm="linf",
        clean_success_rate=clean.success_rate,
        adversarial_success_rate=attacked.success_rate,
        n_episodes=n_episodes,
        min_success_rate=min_success_rate,
        passed=attacked.success_rate >= min_success_rate,
        notes=notes,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
    )


class _StopTrainingOnMaxSeconds(BaseCallback):
    """Halt SB3 training once a wall-clock budget elapses."""

    def __init__(self, max_seconds: float):
        super().__init__()
        self.max_seconds = max_seconds
        self._start = None

    def _on_training_start(self) -> None:
        self._start = time.time()

    def _on_step(self) -> bool:
        return (time.time() - self._start) < self.max_seconds


def train_adversary(
    victim: Policy,
    env: gym.Env,
    epsilon: float,
    timesteps: int = 20_000,
    seed: int = 0,
    verbose: int = 0,
    target_key: str = DEFAULT_TARGET_KEY,
    max_seconds: float | None = None,
):
    """Train a PPO adversary against the frozen victim.

    Returns an adversary usable with :func:`run_adversarial_test`.
    ``max_seconds`` time-boxes training (returning the partially trained
    adversary at the deadline). Raises on failure — use
    :func:`get_adversary` for automatic fallback to the random baseline.
    """
    from stable_baselines3 import PPO

    attack_env = ObservationPerturbationEnv(env, victim, epsilon, target_key=target_key)
    # the adversary observes the full (possibly Dict) observation
    policy_type = (
        "MultiInputPolicy"
        if isinstance(attack_env.observation_space, gym.spaces.Dict)
        else "MlpPolicy"
    )
    model = PPO(
        policy_type,
        attack_env,
        seed=seed,
        policy_kwargs={"net_arch": [32, 32]},
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        verbose=verbose,
        device="cpu",
    )
    callback = _StopTrainingOnMaxSeconds(max_seconds) if max_seconds else None
    model.learn(total_timesteps=timesteps, callback=callback)
    return PPOAdversary(model, epsilon)


def get_adversary(
    victim: Policy,
    env: gym.Env,
    epsilon: float,
    timesteps: int = 20_000,
    seed: int = 0,
    train: bool = True,
    log: Callable[[str], None] = print,
    target_key: str = DEFAULT_TARGET_KEY,
    max_seconds: float | None = None,
):
    """Return (adversary, notes): the trained PPO adversary, or the random
    baseline with an explanatory note if training is disabled or fails."""
    if not train:
        return RandomAdversary(epsilon, seed=seed, target_key=target_key), "learned adversary disabled; random baseline"
    try:
        adversary = train_adversary(
            victim, env, epsilon, timesteps=timesteps, seed=seed,
            target_key=target_key, max_seconds=max_seconds,
        )
        return adversary, ""
    except Exception as exc:  # deliberate blanket catch: never lose the category
        log(f"[cotter] PPO adversary training failed ({exc!r}); falling back to random baseline")
        return (
            RandomAdversary(epsilon, seed=seed, target_key=target_key),
            f"PPO adversary training failed ({type(exc).__name__}); random baseline used",
        )
