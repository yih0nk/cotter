"""Tests for the class-keyed pretrained adversary zoo."""

import gymnasium as gym
import pytest

from cotter.policy import load_policy
from cotter.tests.action_adversarial import train_action_adversary
from cotter.tests.adversarial import train_adversary
from cotter.zoo.pretrained import PretrainedEntry, PretrainedZoo

ENV_ID = "InvertedPendulum-v5"
VICTIM = "artifacts/victim_ppo_inverted_pendulum.zip"


@pytest.fixture
def env():
    e = gym.make(ENV_ID)
    yield e
    e.close()


@pytest.fixture
def victim(env):
    from stable_baselines3 import PPO

    return load_policy(VICTIM, env, algo=PPO)


def tiny_obs_adversary(victim, env):
    return train_adversary(victim, env, epsilon=0.1, timesteps=64, seed=0)


class TestPretrainedZoo:
    def test_empty_zoo(self, tmp_path):
        zoo = PretrainedZoo(tmp_path)
        assert zoo.entries() == []
        assert zoo.lookup("ant", ENV_ID, 0.1) is None
        assert zoo.load("ant", ENV_ID, 0.1) is None

    def test_register_and_lookup(self, tmp_path, victim, env):
        zoo = PretrainedZoo(tmp_path)
        adv = tiny_obs_adversary(victim, env)
        entry = zoo.register(adv, robot_class="pendulum", env_id=ENV_ID, notes="demo")
        assert isinstance(entry, PretrainedEntry)
        assert entry.robot_class == "pendulum"
        assert entry.attack == "observation"
        found = zoo.lookup("pendulum", ENV_ID, 0.1)
        assert found is not None and found.notes == "demo"
        assert (tmp_path / entry.path).exists()

    def test_reload_returns_usable_adversary(self, tmp_path, victim, env):
        zoo = PretrainedZoo(tmp_path)
        zoo.register(tiny_obs_adversary(victim, env), "pendulum", ENV_ID)
        reloaded = zoo.load("pendulum", ENV_ID, 0.1)
        assert reloaded is not None
        delta = reloaded.perturb(env.observation_space.sample())
        assert delta.shape == env.observation_space.shape

    def test_entries_filter_by_class(self, tmp_path, victim, env):
        zoo = PretrainedZoo(tmp_path)
        zoo.register(tiny_obs_adversary(victim, env), "pendulum", ENV_ID)
        assert len(zoo.entries("pendulum")) == 1
        assert zoo.entries("other") == []

    def test_register_replaces_same_key(self, tmp_path, victim, env):
        zoo = PretrainedZoo(tmp_path)
        zoo.register(tiny_obs_adversary(victim, env), "pendulum", ENV_ID, notes="v1")
        zoo.register(tiny_obs_adversary(victim, env), "pendulum", ENV_ID, notes="v2")
        entries = zoo.entries("pendulum")
        assert len(entries) == 1 and entries[0].notes == "v2"

    def test_action_attack_roundtrip(self, tmp_path, victim, env):
        zoo = PretrainedZoo(tmp_path)
        adv = train_action_adversary(victim, env, epsilon=0.5, timesteps=64, seed=0)
        zoo.register(adv, "pendulum", ENV_ID, attack="action")
        reloaded = zoo.load("pendulum", ENV_ID, 0.5, attack="action")
        assert reloaded.name == "action_ppo"

    def test_prune_removes_missing_artifacts(self, tmp_path, victim, env):
        zoo = PretrainedZoo(tmp_path)
        entry = zoo.register(tiny_obs_adversary(victim, env), "pendulum", ENV_ID)
        (tmp_path / entry.path).unlink()  # delete the artifact behind the index
        removed = zoo.prune()
        assert len(removed) == 1
        assert zoo.entries() == []

    def test_register_rejects_bad_attack(self, tmp_path, victim, env):
        zoo = PretrainedZoo(tmp_path)
        with pytest.raises(ValueError, match="attack"):
            zoo.register(tiny_obs_adversary(victim, env), "pendulum", ENV_ID, attack="weights")

    def test_register_rejects_non_adversary(self, tmp_path):
        zoo = PretrainedZoo(tmp_path)
        with pytest.raises(TypeError):
            zoo.register(object(), "pendulum", ENV_ID)
