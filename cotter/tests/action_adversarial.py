"""Adversarial robustness testing via bounded ACTION perturbations.

Where the observation-perturbation attack (:mod:`cotter.tests.adversarial`)
corrupts what the victim *sees*, this attacks what the victim *does*: at
every timestep a bounded delta (``||delta||_inf <= epsilon``, in action
units) is added to the victim's action before it reaches the actuators,
then clipped back into the action space. This models actuator noise,
control-channel corruption, and degraded/faulty actuators — a threat
model distinct from sensor attacks.

The victim's weights are never modified. Two adversaries mirror the
observation attack: :class:`RandomActionAdversary` (the guaranteed floor)
and a PPO adversary trained on :class:`ActionPerturbationEnv`.
"""

from __future__ import annotations

from typing import Protocol

import gymnasium as gym
import numpy as np

from cotter.policy import Policy


class ActionAdversary(Protocol):
    name: str

    def perturb(self, obs, action) -> np.ndarray:
        """Return a delta for the victim's ``action`` (``||delta||_inf <= epsilon``)."""
        ...


def require_box_action(action_space: gym.Space):
    """Return the action-space shape, or raise if it is not a Box."""
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(
            f"action-perturbation testing requires a Box action space; got "
            f"{type(action_space).__name__}"
        )
    return action_space.shape


def clip_to_space(action, action_space: gym.spaces.Box) -> np.ndarray:
    """Clip an action into the space's [low, high] bounds."""
    return np.clip(np.asarray(action, dtype=float), action_space.low, action_space.high)


class RandomActionAdversary:
    """Uniform i.i.d. noise within the L-inf action budget. The floor."""

    def __init__(self, epsilon: float, action_shape, seed: int = 0) -> None:
        self.epsilon = epsilon
        self.name = "action_random"
        self._shape = action_shape
        self._rng = np.random.default_rng(seed)

    def perturb(self, obs, action) -> np.ndarray:
        return self._rng.uniform(-self.epsilon, self.epsilon, size=self._shape)


class NullActionAdversary:
    """Zero perturbation; useful for sanity checks."""

    def __init__(self) -> None:
        self.name = "action_null"

    def perturb(self, obs, action) -> np.ndarray:
        return np.zeros_like(np.asarray(action, dtype=float))


class PPOActionAdversary:
    """A trained SB3 model whose output is the action-space perturbation.

    The model observes the environment observation and emits a vector in
    [-1, 1]^k, scaled into the epsilon budget.
    """

    def __init__(self, model, epsilon: float) -> None:
        self.name = "action_ppo"
        self.epsilon = epsilon
        self.model = model

    def perturb(self, obs, action) -> np.ndarray:
        delta, _ = self.model.predict(obs, deterministic=True)
        return np.clip(np.asarray(delta, dtype=float), -1.0, 1.0) * self.epsilon


class ActionPerturbationEnv(gym.Env):
    """The action attack phrased as an RL environment for the adversary.

    Observation: the true environment observation. Action: a vector in
    [-1, 1]^k, scaled by epsilon, added to the frozen victim's action and
    clipped to the action space. Reward: negative of the victim's reward,
    so maximizing adversary return minimizes victim performance.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env, victim: Policy, epsilon: float) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive; got {epsilon}")
        shape = require_box_action(env.action_space)
        self.env = env
        self.victim = victim
        self.epsilon = epsilon
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32)
        self._true_obs = None

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._true_obs = obs
        return obs, info

    def step(self, action):
        delta = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.epsilon
        victim_action = np.asarray(self.victim.predict(self._true_obs), dtype=float)
        corrupted = clip_to_space(victim_action + delta, self.env.action_space)
        obs, reward, terminated, truncated, info = self.env.step(corrupted)
        self._true_obs = obs
        return obs, -float(reward), terminated, truncated, info

    def close(self):
        self.env.close()


class ActionPerturbedPolicy:
    """The victim seen through the action adversary: perturb the action.

    Runs the full attack loop through the ordinary rollout runner, so the
    adversarial evaluation records the same instrumented step infos as
    every other category.
    """

    def __init__(self, victim: Policy, adversary: ActionAdversary, action_space: gym.spaces.Box):
        self.victim = victim
        self.adversary = adversary
        self.action_space = action_space
        self.name = f"{victim.name}+{adversary.name}"

    def predict(self, obs) -> np.ndarray:
        action = np.asarray(self.victim.predict(obs), dtype=float)
        delta = self.adversary.perturb(obs, action)
        return clip_to_space(action + delta, self.action_space)
