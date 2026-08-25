"""Tests for action-perturbation adversarial testing.

Integration tests against a real InvertedPendulum-v5 env with a trivial
constant policy — the attack machinery is what's under test, not policy
quality.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from cotter.policy import load_policy
from cotter.tests.action_adversarial import (
    ActionPerturbationEnv,
    ActionPerturbedPolicy,
    NullActionAdversary,
    RandomActionAdversary,
    clip_to_space,
    require_box_action,
)

ENV_ID = "InvertedPendulum-v5"


@pytest.fixture
def env():
    e = gym.make(ENV_ID)
    yield e
    e.close()


class ConstPolicy(torch.nn.Module):
    """obs(4) -> action(1), constant zero (deterministic)."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class TestHelpers:
    def test_require_box_action_ok(self, env):
        assert require_box_action(env.action_space) == env.action_space.shape

    def test_require_box_action_rejects_discrete(self):
        with pytest.raises(TypeError):
            require_box_action(gym.spaces.Discrete(3))

    def test_clip_to_space_bounds(self, env):
        low, high = env.action_space.low, env.action_space.high
        clipped = clip_to_space(high + 100.0, env.action_space)
        assert np.all(clipped <= high) and np.all(clipped >= low)


class TestAdversaries:
    def test_random_delta_within_budget_and_shape(self, env):
        adv = RandomActionAdversary(0.5, env.action_space.shape, seed=0)
        delta = adv.perturb(env.observation_space.sample(), np.zeros(env.action_space.shape))
        assert delta.shape == env.action_space.shape
        assert np.all(np.abs(delta) <= 0.5 + 1e-9)
        assert adv.name == "action_random"

    def test_null_is_zero(self, env):
        adv = NullActionAdversary()
        delta = adv.perturb(None, np.array([1.0]))
        assert np.all(delta == 0.0)


class TestPerturbedPolicy:
    def test_output_stays_in_action_space(self, env):
        victim = load_policy(ConstPolicy(), env)
        adv = RandomActionAdversary(100.0, env.action_space.shape, seed=1)  # huge budget
        attacked = ActionPerturbedPolicy(victim, adv, env.action_space)
        out = attacked.predict(env.observation_space.sample())
        assert np.all(out <= env.action_space.high) and np.all(out >= env.action_space.low)


class TestActionPerturbationEnv:
    def test_reward_is_negated_victim_reward(self, env):
        victim = load_policy(ConstPolicy(), env)
        attack_env = ActionPerturbationEnv(env, victim, epsilon=0.5)
        assert attack_env.action_space.shape == env.action_space.shape
        attack_env.reset(seed=0)
        _, reward, _, _, _ = attack_env.step(attack_env.action_space.sample())
        # InvertedPendulum pays +1 per surviving step, so the negated
        # adversary reward is non-positive.
        assert reward <= 0.0

    def test_rejects_positive_epsilon_only(self, env):
        victim = load_policy(ConstPolicy(), env)
        with pytest.raises(ValueError):
            ActionPerturbationEnv(env, victim, epsilon=0.0)
