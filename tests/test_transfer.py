"""Tests for transfer attacks (pretrained adversary -> fresh victim)."""

import gymnasium as gym
import pytest
import torch

from cotter.policy import load_policy
from cotter.tests.adversarial import train_adversary
from cotter.tests.action_adversarial import train_action_adversary
from cotter.tests.transfer import (
    IncompatibleAdversaryError,
    check_adversary_compatible,
    transfer_attack,
)

ENV_ID = "InvertedPendulum-v5"
VICTIM = "artifacts/victim_ppo_inverted_pendulum.zip"


@pytest.fixture
def env():
    e = gym.make(ENV_ID)
    yield e
    e.close()


@pytest.fixture
def reference_victim(env):
    from stable_baselines3 import PPO

    return load_policy(VICTIM, env, algo=PPO)


class FreshPolicy(torch.nn.Module):
    """A different victim (obs4 -> action1) than the reference."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def survival(total_reward, length, terminated, truncated, final_info):
    return length >= 20


class TestCompatibility:
    def test_matching_adversary_is_compatible(self, env, reference_victim):
        adv = train_adversary(reference_victim, env, epsilon=0.1, timesteps=64, seed=0)
        check_adversary_compatible(adv, env)  # must not raise

    def test_wrong_env_is_incompatible(self, reference_victim):
        train_env = gym.make(ENV_ID)
        adv = train_adversary(reference_victim, train_env, epsilon=0.1, timesteps=64)
        train_env.close()
        other = gym.make("Reacher-v5")  # different obs/action dims
        try:
            with pytest.raises(IncompatibleAdversaryError):
                check_adversary_compatible(adv, other)
        finally:
            other.close()


class TestTransferAttack:
    def test_transfer_to_fresh_victim(self, env, reference_victim):
        # train against the reference victim, attack a DIFFERENT victim
        adv = train_adversary(reference_victim, env, epsilon=0.1, timesteps=64, seed=0)
        fresh = load_policy(FreshPolicy(), env)
        result = transfer_attack(
            adv, fresh, env, survival, n_episodes=4, robot_class="pendulum"
        )
        assert "pretrained transfer attack" in result.notes
        assert "class=pendulum" in result.notes
        assert 0.0 <= result.adversarial_success_rate <= 1.0

    def test_epsilon_defaults_to_adversary_budget(self, env, reference_victim):
        adv = train_adversary(reference_victim, env, epsilon=0.1, timesteps=64, seed=0)
        result = transfer_attack(adv, reference_victim, env, survival, n_episodes=3)
        assert result.epsilon == 0.1

    def test_action_transfer(self, env, reference_victim):
        adv = train_action_adversary(reference_victim, env, epsilon=0.5, timesteps=64, seed=0)
        fresh = load_policy(FreshPolicy(), env)
        result = transfer_attack(
            adv, fresh, env, survival, attack="action", n_episodes=4
        )
        assert result.adversary_type == "action_ppo"

    def test_incompatible_adversary_rejected(self, env, reference_victim):
        other = gym.make("Reacher-v5")
        adv = train_adversary(
            load_policy(VICTIM, gym.make(ENV_ID), algo=__import__("stable_baselines3").PPO),
            gym.make(ENV_ID), epsilon=0.1, timesteps=64,
        )
        try:
            with pytest.raises(IncompatibleAdversaryError):
                transfer_attack(adv, reference_victim, other, survival, n_episodes=2)
        finally:
            other.close()
