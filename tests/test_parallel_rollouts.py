"""Parallel rollouts must be bit-identical to serial on the same seeds."""

import gymnasium as gym
import pytest

from cotter.envs.factory import WrappedEnvFactory
from cotter.envs.wrapper import CotterWrapper
from cotter.policy import load_policy
from cotter.runner import (
    make_seed_sequence,
    run_rollouts,
    run_rollouts_parallel,
)
from cotter.tests.safety import SafetyLimit, evaluate_safety
from stable_baselines3 import PPO
from pathlib import Path

VICTIM = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"
ENV_ID = "InvertedPendulum-v5"

pytestmark = pytest.mark.skipif(not VICTIM.exists(), reason="victim artifact missing")


def success(total_reward, length, terminated, truncated, final_info):
    return length >= 100


@pytest.fixture
def victim():
    env = CotterWrapper(gym.make(ENV_ID))
    policy = load_policy(VICTIM, env, algo=PPO, name="victim")
    yield policy
    env.close()


def _serial(victim, seeds, record_infos=True):
    env = CotterWrapper(gym.make(ENV_ID))
    try:
        return run_rollouts(victim, env, len(seeds), success, seeds=seeds, record_infos=record_infos)
    finally:
        env.close()


class TestParallelMatchesSerial:
    def test_identical_scalar_outputs(self, victim):
        seeds = make_seed_sequence(6, base_seed=42)
        serial = _serial(victim, seeds, record_infos=False)
        parallel = run_rollouts_parallel(
            victim, WrappedEnvFactory(ENV_ID), len(seeds), success,
            seeds=seeds, record_infos=False, n_workers=3,
        )
        assert parallel.seeds == serial.seeds
        assert parallel.lengths == serial.lengths
        assert parallel.returns == serial.returns  # exact float equality
        assert parallel.successes == serial.successes

    def test_ragged_chunk_last_partial(self, victim):
        # 5 episodes over 4 workers -> chunks of 4 then 1
        seeds = make_seed_sequence(5, base_seed=7)
        serial = _serial(victim, seeds, record_infos=False)
        parallel = run_rollouts_parallel(
            victim, WrappedEnvFactory(ENV_ID), 5, success,
            seeds=seeds, record_infos=False, n_workers=4,
        )
        assert parallel.lengths == serial.lengths
        assert parallel.returns == serial.returns

    def test_recorded_infos_yield_identical_safety(self, victim):
        seeds = make_seed_sequence(4, base_seed=13)
        limits = [SafetyLimit("cotter/joint_velocities", 5.0),
                  SafetyLimit("cotter/actuator_forces", 2.5)]
        serial = _serial(victim, seeds, record_infos=True)
        parallel = run_rollouts_parallel(
            victim, WrappedEnvFactory(ENV_ID), 4, success,
            seeds=seeds, record_infos=True, n_workers=2,
        )
        s_res = evaluate_safety(serial.episode_infos, limits)
        p_res = evaluate_safety(parallel.episode_infos, limits)
        assert p_res.decision == s_res.decision
        assert p_res.n_timesteps_checked == s_res.n_timesteps_checked
        assert p_res.worst_observed == s_res.worst_observed  # exact

    def test_n_workers_one_falls_back_to_serial(self, victim):
        seeds = make_seed_sequence(3, base_seed=1)
        serial = _serial(victim, seeds, record_infos=False)
        parallel = run_rollouts_parallel(
            victim, WrappedEnvFactory(ENV_ID), 3, success,
            seeds=seeds, record_infos=False, n_workers=1,
        )
        assert parallel.returns == serial.returns
        assert parallel.lengths == serial.lengths


class TestValidation:
    def test_too_few_seeds_rejected(self, victim):
        with pytest.raises(ValueError, match="need 5 seeds"):
            run_rollouts_parallel(
                victim, WrappedEnvFactory(ENV_ID), 5, success,
                seeds=[1, 2], n_workers=2,
            )

    def test_zero_workers_rejected(self, victim):
        with pytest.raises(ValueError, match="n_workers must be"):
            run_rollouts_parallel(
                victim, WrappedEnvFactory(ENV_ID), 2, success,
                seeds=[1, 2], n_workers=0,
            )
