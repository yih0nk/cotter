"""Tests for `cotter pretrained list` and `cotter pretrained prune`."""

from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from cotter.cli import main
from cotter.policy import load_policy
from cotter.tests.adversarial import train_adversary
from cotter.zoo.pretrained import PretrainedZoo

VICTIM = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"
ENV_ID = "InvertedPendulum-v5"

pytestmark = pytest.mark.skipif(not VICTIM.exists(), reason="victim artifact missing")


@pytest.fixture
def populated(tmp_path):
    env = gym.make(ENV_ID)
    victim = load_policy(VICTIM, env, algo=PPO)
    adv = train_adversary(victim, env, epsilon=0.1, timesteps=64, seed=0)
    entry = PretrainedZoo(tmp_path).register(adv, "pendulum", ENV_ID, notes="demo")
    env.close()
    return tmp_path, entry


class TestPretrainedList:
    def test_empty(self, tmp_path, capsys):
        rc = main(["pretrained", "--root", str(tmp_path), "list"])
        assert rc == 0
        assert "no pretrained adversaries" in capsys.readouterr().out

    def test_lists_entry(self, populated, capsys):
        root, _ = populated
        rc = main(["pretrained", "--root", str(root), "list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "pendulum" in out and ENV_ID in out and "observation" in out

    def test_filter_by_class(self, populated, capsys):
        root, _ = populated
        rc = main(["pretrained", "--root", str(root), "list", "--robot-class", "other"])
        assert rc == 0
        assert "no pretrained adversaries for class 'other'" in capsys.readouterr().out


class TestPretrainedPrune:
    def test_prune_missing(self, populated, capsys):
        root, entry = populated
        (root / entry.path).unlink()
        rc = main(["pretrained", "--root", str(root), "prune"])
        assert rc == 0
        assert "pruned 1 entry" in capsys.readouterr().out

    def test_prune_nothing(self, populated, capsys):
        root, _ = populated
        rc = main(["pretrained", "--root", str(root), "prune"])
        assert rc == 0
        assert "nothing to prune" in capsys.readouterr().out
