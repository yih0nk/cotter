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

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import gymnasium as gym
import numpy as np

from cotter.policy import Policy
from cotter.runner import SuccessFn, make_seed_sequence, run_rollouts


class Adversary(Protocol):
    name: str

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        """Return delta with ||delta||_inf <= epsilon, same shape as obs."""
        ...


class RandomAdversary:
    """Uniform i.i.d. noise within the L-inf budget. The guaranteed floor."""

    def __init__(self, epsilon: float, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.name = "random"
        self._rng = np.random.default_rng(seed)

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-self.epsilon, self.epsilon, size=np.shape(obs))


class NullAdversary:
    """Zero perturbation; useful for sanity checks."""

    def __init__(self) -> None:
        self.name = "null"

    def perturb(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(obs, dtype=float))


class ObservationPerturbationEnv(gym.Env):
    """The attack problem phrased as an RL environment for the adversary.

    Observation: the true environment observation. Action: a vector in
    [-1, 1]^obs_dim, scaled by epsilon and added to what the frozen victim
    sees. Reward: negative of the victim's reward, so maximizing adversary
    return minimizes victim performance.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, victim: Policy, epsilon: float) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive; got {epsilon}")
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError("ObservationPerturbationEnv requires a Box observation space")
        self.env = env
        self.victim = victim
        self.epsilon = epsilon
        self.observation_space = obs_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=obs_space.shape, dtype=np.float32)
        self._true_obs: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._true_obs = obs
        return obs, info

    def step(self, action):
        delta = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.epsilon
        perturbed = self._true_obs + delta
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

    def __init__(self, victim: Policy, adversary: Adversary) -> None:
        self.victim = victim
        self.adversary = adversary
        self.name = f"{victim.name}+{adversary.name}"

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.victim.predict(obs + self.adversary.perturb(obs))


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

    @property
    def success_rate_drop(self) -> float:
        return self.clean_success_rate - self.adversarial_success_rate

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        text = (
            f"{verdict}: success rate {self.clean_success_rate:.1%} clean -> "
            f"{self.adversarial_success_rate:.1%} under {self.adversary_type} "
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
) -> AdversarialResult:
    """Measure worst-case success rate under bounded observation attack.

    Clean and perturbed rollouts share the same seed sequence, so the
    reported drop is a matched comparison. PASS requires the perturbed
    success rate to stay at or above ``min_success_rate``.
    """
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            f"adversarial testing currently supports Box observation spaces "
            f"only; {getattr(env, 'spec', None) and env.spec.id or 'env'} has "
            f"{type(env.observation_space).__name__}. Dict-obs perturbation "
            "is not implemented yet — drop the adversarial section for this env."
        )
    if adversary is None:
        adversary = RandomAdversary(epsilon, seed=base_seed)
    if seeds is None:
        seeds = make_seed_sequence(n_episodes, base_seed)

    clean = run_rollouts(victim, env, n_episodes, success_fn, seeds=seeds, record_infos=False)
    attacked = run_rollouts(
        _PerturbedPolicy(victim, adversary), env, n_episodes, success_fn,
        seeds=seeds, record_infos=False,
    )

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
    )


def train_adversary(
    victim: Policy,
    env: gym.Env,
    epsilon: float,
    timesteps: int = 20_000,
    seed: int = 0,
    verbose: int = 0,
):
    """Train a PPO adversary against the frozen victim.

    Returns an adversary usable with :func:`run_adversarial_test`.
    Raises on failure — use :func:`get_adversary` for automatic fallback
    to the random baseline.
    """
    from stable_baselines3 import PPO

    attack_env = ObservationPerturbationEnv(env, victim, epsilon)
    model = PPO(
        "MlpPolicy",
        attack_env,
        seed=seed,
        policy_kwargs={"net_arch": [32, 32]},
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        verbose=verbose,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps)

    class PPOAdversary:
        def __init__(self) -> None:
            self.name = "ppo"
            self.epsilon = epsilon
            self.model = model

        def perturb(self, obs: np.ndarray) -> np.ndarray:
            action, _ = self.model.predict(obs, deterministic=True)
            return np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * epsilon

    return PPOAdversary()


def get_adversary(
    victim: Policy,
    env: gym.Env,
    epsilon: float,
    timesteps: int = 20_000,
    seed: int = 0,
    train: bool = True,
    log: Callable[[str], None] = print,
):
    """Return (adversary, notes): the trained PPO adversary, or the random
    baseline with an explanatory note if training is disabled or fails."""
    if not train:
        return RandomAdversary(epsilon, seed=seed), "learned adversary disabled; random baseline"
    try:
        return train_adversary(victim, env, epsilon, timesteps=timesteps, seed=seed), ""
    except Exception as exc:  # deliberate blanket catch: never lose the category
        log(f"[cotter] PPO adversary training failed ({exc!r}); falling back to random baseline")
        return (
            RandomAdversary(epsilon, seed=seed),
            f"PPO adversary training failed ({type(exc).__name__}); random baseline used",
        )
