"""Integration tests for the config-driven pipeline.

Runs the real checked-in victim policy on real MuJoCo envs with tiny
episode budgets (success thresholds far below the horizon keep episodes
long but the counts small).
"""

from pathlib import Path

import pytest

from cotter.backends import BackendFactory, GymnasiumBackend
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

    def test_make_env_routes_through_named_backend(self):
        # A test-double backend proves make_env dispatches via from_name.
        calls = []

        class SpyBackend(BackendFactory):
            backend_name = "spy-backend"

            def make_env(self, env_id):
                calls.append(env_id)
                return GymnasiumBackend().make_env(env_id)

        try:
            env = make_env("InvertedPendulum-v5", backend="spy-backend", log=lambda m: None)
            assert calls == ["InvertedPendulum-v5"]
            env.close()
        finally:
            BackendFactory._registry.pop("spy-backend", None)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown backend"):
            make_env("InvertedPendulum-v5", backend="nope", log=lambda m: None)


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

    def test_action_attack_surface(self):
        cfg = parse_config({
            "env": "InvertedPendulum-v5",
            "success": {"type": "min_length", "value": 10},
            "adversarial": {
                "epsilon": 0.5, "n_episodes": 2, "train": False, "attack": "action"
            },
        })
        report = run_from_config(VICTIM, cfg, log=lambda m: None)
        categories = [(r.category, r.name) for r in report.results]
        # action attack produces the action baseline, not the observation one
        assert ("adversarial", "action_random_baseline") in categories
        assert ("adversarial", "random_baseline") not in categories

    def test_both_attack_surfaces(self):
        cfg = parse_config({
            "env": "InvertedPendulum-v5",
            "success": {"type": "min_length", "value": 10},
            "adversarial": {
                "epsilon": 0.5, "n_episodes": 2, "train": False, "attack": "both"
            },
        })
        report = run_from_config(VICTIM, cfg, log=lambda m: None)
        names = [r.name for r in report.results]
        assert "random_baseline" in names
        assert "action_random_baseline" in names

    def test_pretrained_transfer_attack(self, tmp_path):
        import gymnasium as gym

        from cotter.policy import load_policy
        from cotter.tests.adversarial import train_adversary
        from cotter.zoo.pretrained import PretrainedZoo
        from stable_baselines3 import PPO

        # register a pretrained adversary for the "pendulum" class
        train_env = gym.make("InvertedPendulum-v5")
        victim = load_policy(str(VICTIM), train_env, algo=PPO)
        adv = train_adversary(victim, train_env, epsilon=0.1, timesteps=64, seed=0)
        PretrainedZoo(tmp_path).register(adv, "pendulum", "InvertedPendulum-v5")
        train_env.close()

        cfg = parse_config({
            "env": "InvertedPendulum-v5",
            "success": {"type": "min_length", "value": 10},
            "adversarial": {
                "epsilon": 0.1, "n_episodes": 2, "train": False,
                "pretrained": "pendulum", "pretrained_root": str(tmp_path),
            },
        })
        report = run_from_config(VICTIM, cfg, log=lambda m: None)
        names = [r.name for r in report.results]
        assert "pretrained_pendulum_observation" in names
        entry = next(r for r in report.results if r.name == "pretrained_pendulum_observation")
        assert "pretrained transfer attack" in entry.data["notes"]

    def test_pretrained_missing_class_is_skipped(self, tmp_path):
        cfg = parse_config({
            "env": "InvertedPendulum-v5",
            "success": {"type": "min_length", "value": 10},
            "adversarial": {
                "epsilon": 0.1, "n_episodes": 2, "train": False,
                "pretrained": "nonexistent", "pretrained_root": str(tmp_path),
            },
        })
        report = run_from_config(VICTIM, cfg, log=lambda m: None)
        # no pretrained entry, and the run still completes (random baseline present)
        assert not any(r.name.startswith("pretrained_") for r in report.results)
        assert any(r.name == "random_baseline" for r in report.results)

    def test_iso_ts_15066_pfl_category(self):
        cfg = parse_config({
            "env": "InvertedPendulum-v5",
            "success": {"type": "min_length", "value": 10},
            "iso_ts_15066": {
                "m_robot": 5.0,
                # the CotterWrapper exposes joint velocities; used here as the
                # TCP-speed stand-in to exercise the check end-to-end.
                "speed_key": "cotter/joint_velocities",
                "n_episodes": 2,
                "regions": ["chest", "skull_forehead"],
            },
        })
        report = run_from_config(VICTIM, cfg, log=lambda m: None)
        pfl = next((r for r in report.results if r.name == "iso_ts_15066_pfl"), None)
        assert pfl is not None
        assert pfl.category == "safety"
        assert pfl.data["binding_region"] in ("chest", "skull_forehead")
        assert len(pfl.data["region_limits"]) == 2
        assert "ISO/TS 15066" in pfl.summary

    def test_parallel_workers_match_serial_report(self):
        # Safety + regression with n_workers > 1 must produce the same
        # numbers as the serial (default) config on shared base_seed.
        spec = {
            "env": "InvertedPendulum-v5",
            "success": {"type": "min_length", "value": 50},
            "base_seed": 0,
            "safety": {"n_episodes": 4, "limits": {"cotter/joint_velocities": 5.0}},
            "regression": {"baseline": str(VICTIM), "n_pairs": 4},
        }
        serial = run_from_config(VICTIM, parse_config(dict(spec)), log=lambda m: None)
        par_spec = dict(spec)
        par_spec["safety"] = dict(spec["safety"], n_workers=2)
        par_spec["regression"] = dict(spec["regression"], n_workers=2)
        parallel = run_from_config(VICTIM, parse_config(par_spec), log=lambda m: None)

        def by_name(report):
            return {r.name: r for r in report.results}

        s, p = by_name(serial), by_name(parallel)
        assert p["hard_limits"].data["worst_observed"] == s["hard_limits"].data["worst_observed"]
        assert p["hard_limits"].data["n_timesteps_checked"] == s["hard_limits"].data["n_timesteps_checked"]
        assert p["success_mcnemar"].data["p_value"] == s["success_mcnemar"].data["p_value"]
        assert p["return_wilcoxon"].data["p_value"] == s["return_wilcoxon"].data["p_value"]

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

    def test_zoo_trains_then_reuses_adversary(self, tmp_path):
        # First run trains and caches; second run must reuse (no retrain).
        base = {
            "env": "InvertedPendulum-v5",
            "success": {"type": "min_length", "value": 50},
            "adversarial": {
                "epsilon": 0.07, "n_episodes": 2, "train": True,
                "timesteps": 512, "use_zoo": True, "zoo_root": str(tmp_path / "zoo"),
            },
        }
        logs1: list[str] = []
        run_from_config(VICTIM, parse_config(dict(base)), log=logs1.append)
        assert any("training and storing in zoo" in m for m in logs1)

        logs2: list[str] = []
        run_from_config(VICTIM, parse_config(dict(base)), log=logs2.append)
        assert any("reusing cached adversary" in m for m in logs2)
        assert not any("training and storing" in m for m in logs2)

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
