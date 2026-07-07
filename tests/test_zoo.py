"""Tests for the adversary zoo registry."""

from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from cotter.envs.wrapper import CotterWrapper
from cotter.policy import load_policy
from cotter.tests.adversarial import PPOAdversary, RandomAdversary, train_adversary
from cotter.zoo import AdversaryZoo, victim_hash

VICTIM = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"
ENV_ID = "InvertedPendulum-v5"


@pytest.fixture
def env():
    e = CotterWrapper(gym.make(ENV_ID))
    yield e
    e.close()


@pytest.fixture
def victim(env):
    return load_policy(VICTIM, env, algo=PPO, name="victim")


@pytest.fixture
def cheap_adversary(env, victim):
    # tiny training budget: exercises save/load, not attack quality
    return train_adversary(victim, env, epsilon=0.1, timesteps=512, seed=0)


class TestVictimHash:
    def test_path_hash_is_stable(self):
        assert victim_hash(VICTIM) == victim_hash(VICTIM)

    def test_path_and_loaded_policy_hash_differently_but_stably(self, victim):
        # path hashes file bytes; loaded policy hashes params — both stable,
        # not required to be equal, but each reproducible.
        assert victim_hash(victim) == victim_hash(victim)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            victim_hash(Path("no_such_policy.zip"))

    def test_unhashable_type_raises(self):
        with pytest.raises(TypeError):
            victim_hash(42)


class TestSaveLoadLookup:
    def test_miss_before_save(self, tmp_path):
        zoo = AdversaryZoo(tmp_path)
        assert zoo.lookup(ENV_ID, VICTIM, 0.1) is None
        assert zoo.load(ENV_ID, VICTIM, 0.1) is None

    def test_save_then_load_roundtrip(self, tmp_path, cheap_adversary):
        zoo = AdversaryZoo(tmp_path)
        entry = zoo.save(cheap_adversary, ENV_ID, VICTIM, notes="test")
        assert entry.key() == (ENV_ID, victim_hash(VICTIM), 0.1)
        assert (tmp_path / entry.path).exists()

        reloaded = zoo.load(ENV_ID, VICTIM, 0.1)
        assert isinstance(reloaded, PPOAdversary)
        assert reloaded.epsilon == 0.1

    def test_reloaded_adversary_matches_original(self, tmp_path, cheap_adversary, env):
        zoo = AdversaryZoo(tmp_path)
        zoo.save(cheap_adversary, ENV_ID, VICTIM)
        reloaded = zoo.load(ENV_ID, VICTIM, 0.1)
        obs, _ = env.reset(seed=0)
        # deterministic prediction: identical perturbation after reload
        assert (cheap_adversary.perturb(obs) == reloaded.perturb(obs)).all()

    def test_epsilon_is_part_of_key(self, tmp_path, cheap_adversary):
        zoo = AdversaryZoo(tmp_path)
        zoo.save(cheap_adversary, ENV_ID, VICTIM)
        assert zoo.lookup(ENV_ID, VICTIM, 0.2) is None  # different budget: miss
        assert zoo.lookup(ENV_ID, VICTIM, 0.1) is not None

    def test_env_id_is_part_of_key(self, tmp_path, cheap_adversary):
        zoo = AdversaryZoo(tmp_path)
        zoo.save(cheap_adversary, ENV_ID, VICTIM)
        assert zoo.lookup("OtherEnv-v0", VICTIM, 0.1) is None

    def test_resave_replaces_entry(self, tmp_path, cheap_adversary):
        zoo = AdversaryZoo(tmp_path)
        zoo.save(cheap_adversary, ENV_ID, VICTIM, notes="first")
        zoo.save(cheap_adversary, ENV_ID, VICTIM, notes="second")
        entries = zoo.entries(ENV_ID)
        assert len(entries) == 1
        assert entries[0].notes == "second"

    def test_entries_filter_by_env(self, tmp_path, cheap_adversary):
        zoo = AdversaryZoo(tmp_path)
        zoo.save(cheap_adversary, ENV_ID, VICTIM)
        assert len(zoo.entries()) == 1
        assert len(zoo.entries(ENV_ID)) == 1
        assert zoo.entries("Nope-v0") == []


class TestGuards:
    def test_random_adversary_not_storable(self, tmp_path):
        zoo = AdversaryZoo(tmp_path)
        with pytest.raises(TypeError, match="PPOAdversary"):
            zoo.save(RandomAdversary(0.1), ENV_ID, VICTIM)

    def test_load_with_deleted_artifact_raises(self, tmp_path, cheap_adversary):
        zoo = AdversaryZoo(tmp_path)
        entry = zoo.save(cheap_adversary, ENV_ID, VICTIM)
        (tmp_path / entry.path).unlink()
        with pytest.raises(FileNotFoundError, match="modified outside"):
            zoo.load(ENV_ID, VICTIM, 0.1)

    def test_index_survives_reinstantiation(self, tmp_path, cheap_adversary):
        AdversaryZoo(tmp_path).save(cheap_adversary, ENV_ID, VICTIM)
        # a fresh object reading the same root sees the entry
        assert AdversaryZoo(tmp_path).lookup(ENV_ID, VICTIM, 0.1) is not None
