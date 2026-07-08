"""Tests for `cotter zoo list` and `cotter zoo prune`."""

from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import PPO

from cotter.cli import main
from cotter.envs.wrapper import CotterWrapper
from cotter.policy import load_policy
from cotter.tests.adversarial import train_adversary
from cotter.zoo import AdversaryZoo

VICTIM = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"
ENV_ID = "InvertedPendulum-v5"

pytestmark = pytest.mark.skipif(not VICTIM.exists(), reason="victim artifact missing")


@pytest.fixture
def populated_zoo(tmp_path):
    """A zoo root with one cheaply-trained cached adversary."""
    env = CotterWrapper(gym.make(ENV_ID))
    victim = load_policy(VICTIM, env, algo=PPO)
    adv = train_adversary(victim, env, epsilon=0.1, timesteps=512, seed=0)
    zoo = AdversaryZoo(tmp_path)
    entry = zoo.save(adv, ENV_ID, VICTIM)
    env.close()
    return tmp_path, entry


class TestZooList:
    def test_empty_zoo(self, tmp_path, capsys):
        rc = main(["zoo", "--root", str(tmp_path), "list"])
        assert rc == 0
        assert "no cached adversaries" in capsys.readouterr().out

    def test_lists_entry(self, populated_zoo, capsys):
        root, entry = populated_zoo
        rc = main(["zoo", "--root", str(root), "list"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "1 cached adversary" in out
        assert ENV_ID in out
        assert entry.victim_hash in out

    def test_env_filter_matches(self, populated_zoo, capsys):
        root, _ = populated_zoo
        rc = main(["zoo", "--root", str(root), "list", "--env", ENV_ID])
        assert rc == 0
        assert "1 cached adversary" in capsys.readouterr().out

    def test_env_filter_excludes(self, populated_zoo, capsys):
        root, _ = populated_zoo
        rc = main(["zoo", "--root", str(root), "list", "--env", "OtherEnv-v0"])
        assert rc == 0
        assert "no cached adversaries for env 'OtherEnv-v0'" in capsys.readouterr().out

    def test_missing_artifact_flagged(self, populated_zoo, capsys):
        root, entry = populated_zoo
        (root / entry.path).unlink()
        main(["zoo", "--root", str(root), "list"])
        assert "[MISSING ARTIFACT]" in capsys.readouterr().out


class TestZooPrune:
    def test_prune_nothing(self, populated_zoo, capsys):
        root, _ = populated_zoo
        rc = main(["zoo", "--root", str(root), "prune"])
        assert rc == 0
        assert "nothing to prune" in capsys.readouterr().out

    def test_prune_removes_missing(self, populated_zoo, capsys):
        root, entry = populated_zoo
        (root / entry.path).unlink()
        rc = main(["zoo", "--root", str(root), "prune"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "pruned 1 entry" in out
        # entry gone afterward
        main(["zoo", "--root", str(root), "list"])
        assert "no cached adversaries" in capsys.readouterr().out


class TestUsage:
    def test_zoo_requires_subcommand(self):
        with pytest.raises(SystemExit) as exc:
            main(["zoo"])
        assert exc.value.code == 2
