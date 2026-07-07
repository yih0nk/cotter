"""Tests for the adversarial perturbation test category."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from cotter.envs.wrapper import CotterWrapper
from cotter.policy import load_policy
from cotter.tests.adversarial import (
    NullAdversary,
    ObservationPerturbationEnv,
    RandomAdversary,
    get_adversary,
    run_adversarial_test,
    train_adversary,
)

ENV_ID = "InvertedPendulum-v5"


class ZeroPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def env():
    e = CotterWrapper(gym.make(ENV_ID))
    yield e
    e.close()


@pytest.fixture
def victim(env):
    return load_policy(ZeroPolicy(), env, name="zero-victim")


def survival(total_reward, length, terminated, truncated, final_info):
    return length >= 20


class TestRandomAdversary:
    def test_bounded_by_epsilon(self):
        adv = RandomAdversary(epsilon=0.25, seed=1)
        for _ in range(50):
            delta = adv.perturb(np.zeros(4))
            assert delta.shape == (4,)
            assert np.all(np.abs(delta) <= 0.25)

    def test_deterministic_given_seed(self):
        a = RandomAdversary(0.1, seed=5).perturb(np.zeros(4))
        b = RandomAdversary(0.1, seed=5).perturb(np.zeros(4))
        np.testing.assert_array_equal(a, b)


class TestPerturbationEnv:
    def test_spaces_and_reward_negation(self, env, victim):
        attack_env = ObservationPerturbationEnv(env, victim, epsilon=0.1)
        assert attack_env.observation_space == env.observation_space
        assert attack_env.action_space.shape == env.observation_space.shape
        attack_env.reset(seed=0)
        # InvertedPendulum-v5 pays the victim +1 per surviving step, so the
        # adversary must see -1.
        _, adv_reward, _, _, _ = attack_env.step(np.zeros(4, dtype=np.float32))
        assert adv_reward == -1.0

    def test_episode_terminates(self, env, victim):
        attack_env = ObservationPerturbationEnv(env, victim, epsilon=0.1)
        attack_env.reset(seed=0)
        done = False
        for _ in range(2000):
            _, _, terminated, truncated, _ = attack_env.step(
                attack_env.action_space.sample()
            )
            if terminated or truncated:
                done = True
                break
        assert done

    def test_rejects_nonpositive_epsilon(self, env, victim):
        with pytest.raises(ValueError):
            ObservationPerturbationEnv(env, victim, epsilon=0.0)


class TestRunAdversarialTest:
    def test_null_adversary_matches_clean(self, env, victim):
        # Zero perturbation + deterministic victim + shared seeds must give
        # exactly the clean success rate.
        res = run_adversarial_test(
            victim, env, survival, epsilon=0.05,
            n_episodes=4, adversary=NullAdversary(), base_seed=3,
        )
        assert res.adversarial_success_rate == res.clean_success_rate
        assert res.success_rate_drop == 0.0

    def test_random_adversary_result_structure(self, env, victim):
        res = run_adversarial_test(
            victim, env, survival, epsilon=0.3, n_episodes=4,
            min_success_rate=0.0, base_seed=0,
        )
        assert res.adversary_type == "random"
        assert res.norm == "linf"
        assert 0.0 <= res.adversarial_success_rate <= 1.0
        assert res.passed  # min_success_rate=0 always passes
        d = res.to_dict()
        assert d["epsilon"] == 0.3
        assert "clean_success_rate" in d
        assert "PASS" in res.summary()

    def test_fail_when_below_threshold(self, env, victim):
        res = run_adversarial_test(
            victim, env, lambda *a: False, epsilon=0.05,
            n_episodes=2, min_success_rate=0.5,
        )
        assert not res.passed
        assert "FAIL" in res.summary()


class TestLearnedAdversary:
    def test_train_adversary_smoke(self, env, victim):
        # Minimal-budget training: verifies the PPO wiring end to end, not
        # attack quality.
        adv = train_adversary(victim, env, epsilon=0.1, timesteps=1024, seed=0)
        assert adv.name == "ppo"
        delta = adv.perturb(env.reset(seed=0)[0])
        assert delta.shape == (4,)
        assert np.all(np.abs(delta) <= 0.1 + 1e-9)

    def test_get_adversary_fallback_on_failure(self, env, victim, monkeypatch):
        import cotter.tests.adversarial as mod

        def boom(*args, **kwargs):
            raise RuntimeError("simulated training crash")

        monkeypatch.setattr(mod, "train_adversary", boom)
        messages = []
        adv, notes = mod.get_adversary(
            victim, env, epsilon=0.1, log=messages.append
        )
        assert adv.name == "random"
        assert "falling back" in messages[0]
        assert "random baseline used" in notes

    def test_get_adversary_train_disabled(self, env, victim):
        adv, notes = get_adversary(victim, env, epsilon=0.1, train=False)
        assert adv.name == "random"
        assert "disabled" in notes
