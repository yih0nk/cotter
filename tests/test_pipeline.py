"""Integration tests for the config-driven pipeline.

Runs the real checked-in victim policy on real MuJoCo envs with tiny
episode budgets (success thresholds far below the horizon keep episodes
long but the counts small).
"""

from pathlib import Path

import pytest

from cotter.config import parse_config
from cotter.pipeline import make_env, resolve_algo, run_from_config

VICTIM = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"


class TestResolveAlgo:
    def test_known_algo(self):
        from stable_baselines3 import PPO

        assert resolve_algo("PPO") is PPO

    def test_unknown_algo(self):
        with pytest.raises(ValueError, match="unknown SB3 algorithm 'XQL'"):
            resolve_algo("XQL")


class TestMakeEnv:
    def test_mujoco_env_wrapped(self):
        from cotter.envs.wrapper import CotterWrapper

        env = make_env("InvertedPendulum-v5", log=lambda m: None)
        assert isinstance(env, CotterWrapper)
        env.close()

    def test_non_mujoco_env_passthrough(self):
        from cotter.envs.wrapper import CotterWrapper

        messages = []
        env = make_env("CartPole-v1", log=messages.append)
        assert not isinstance(env, CotterWrapper)
        assert any("not MuJoCo-backed" in m for m in messages)
        env.close()


@pytest.mark.skipif(not VICTIM.exists(), reason="victim artifact missing")
class TestRunFromConfig:
    def test_all_categories_execute(self, tmp_path):
        cfg = parse_config(
            {
                "env": "InvertedPendulum-v5",
                "success": {"type": "min_length", "value": 50},
                "base_seed": 0,
                "performance": {"p0": 0.5, "p1": 0.9, "n_max": 10},
                "safety": {
                    "n_episodes": 2,
                    "limits": {"cotter/joint_velocities": 5.0},
                },
                "regression": {"baseline": str(VICTIM), "n_pairs": 4},
                "adversarial": {"epsilon": 0.07, "n_episodes": 2, "train": False},
                "report": str(tmp_path / "report.json"),
            }
        )
        report = run_from_config(VICTIM, cfg, log=lambda m: None)

        categories = [(r.category, r.name) for r in report.results]
        assert ("performance", "sprt_success_rate") in categories
        assert ("safety", "hard_limits") in categories
        assert ("regression", "success_mcnemar") in categories
        assert ("regression", "return_wilcoxon") in categories
        assert ("adversarial", "random_baseline") in categories
        # train: false must not produce a learned-adversary entry
        assert not any(name.startswith("learned_") for _, name in categories)
        assert (tmp_path / "report.json").exists()

    def test_only_declared_categories_run(self):
        cfg = parse_config(
            {
                "env": "InvertedPendulum-v5",
                "success": {"type": "min_length", "value": 50},
                "performance": {"p0": 0.5, "p1": 0.9, "n_max": 5},
            }
        )
        report = run_from_config(VICTIM, cfg, log=lambda m: None)
        assert [r.category for r in report.results] == ["performance"]

    def test_self_regression_is_clean(self):
        # A policy compared against itself on shared seeds has zero
        # discordant pairs: NO_REGRESSION with p = 1.
        cfg = parse_config(
            {
                "env": "InvertedPendulum-v5",
                "success": {"type": "min_length", "value": 50},
                "regression": {"baseline": str(VICTIM), "n_pairs": 3},
            }
        )
        report = run_from_config(VICTIM, cfg, log=lambda m: None)
        mcnemar = next(r for r in report.results if r.name == "success_mcnemar")
        assert mcnemar.passed is True
        assert mcnemar.data["p_value"] == 1.0

    def test_safety_on_non_mujoco_env_rejected(self):
        cfg = parse_config(
            {
                "env": "CartPole-v1",
                "success": {"type": "min_length", "value": 10},
                "safety": {"limits": {"q": 1.0}},
            }
        )
        with pytest.raises(ValueError, match="not MuJoCo-backed"):
            run_from_config(VICTIM, cfg, log=lambda m: None)
