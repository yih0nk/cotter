"""Integration tests for the wrapper, policy loading, and rollout runner.

These spin up real MuJoCo InvertedPendulum-v5 environments (CPU, tiny
model) — they are integration tests, not mocks.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from cotter.envs.wrapper import (
    ACTUATOR_FORCES,
    CONTACT_COUNT,
    CONTACT_FORCES,
    INSTRUMENTED_KEYS,
    JOINT_VELOCITIES,
    CotterWrapper,
)
from cotter.policy import SpaceMismatchError, TorchPolicy, load_policy, validate_spaces
from cotter.runner import make_seed_sequence, rollout_one, run_rollouts

ENV_ID = "InvertedPendulum-v5"


@pytest.fixture
def env():
    e = CotterWrapper(gym.make(ENV_ID))
    yield e
    e.close()


class ZeroPolicy(torch.nn.Module):
    """Deterministic obs(4) -> action(1) module that always outputs 0."""

    def __init__(self, in_dim=4, out_dim=1):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def survival_success(total_reward, length, terminated, truncated, final_info):
    return length >= 20


class TestCotterWrapper:
    def test_step_and_reset_infos_instrumented(self, env):
        _, reset_info = env.reset(seed=0)
        for key in INSTRUMENTED_KEYS:
            assert key in reset_info
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info[JOINT_VELOCITIES].shape == (2,)  # cart slide + pole hinge
        assert info[ACTUATOR_FORCES].shape == (1,)  # single cart actuator
        assert isinstance(info[CONTACT_COUNT], int)
        # world + cart + pole bodies, one wrench magnitude each, all >= 0
        assert info[CONTACT_FORCES].shape == (3,)
        assert np.all(info[CONTACT_FORCES] >= 0.0)

    def test_values_are_copies_not_views(self, env):
        env.reset(seed=0)
        _, _, _, _, info1 = env.step(np.array([3.0], dtype=np.float32))
        snapshot = info1[JOINT_VELOCITIES].copy()
        env.step(np.array([-3.0], dtype=np.float32))
        np.testing.assert_array_equal(info1[JOINT_VELOCITIES], snapshot)

    def test_rejects_non_mujoco_env(self):
        with pytest.raises(TypeError, match="MuJoCo"):
            CotterWrapper(gym.make("CartPole-v1"))


class TestPolicyLoading:
    def test_torch_policy_validates_and_predicts(self, env):
        policy = load_policy(ZeroPolicy(), env, name="zero")
        action = policy.predict(env.reset(seed=0)[0])
        assert action.shape == env.action_space.shape
        assert np.allclose(action, 0.0)

    def test_torch_pt_file_roundtrip(self, env, tmp_path):
        path = tmp_path / "zero.pt"
        torch.save(ZeroPolicy(), path)
        policy = load_policy(path, env)
        assert policy.name == "zero"
        assert policy.predict(env.reset(seed=0)[0]).shape == (1,)

    def test_wrong_input_dim_fails_loudly(self, env):
        with pytest.raises(SpaceMismatchError, match="sample observation"):
            load_policy(ZeroPolicy(in_dim=7), env)

    def test_wrong_output_shape_fails_loudly(self, env):
        with pytest.raises(SpaceMismatchError, match="expects"):
            load_policy(ZeroPolicy(out_dim=3), env)

    def test_sb3_space_mismatch_fails_loudly(self, env):
        from stable_baselines3 import PPO

        from cotter.policy import SB3Policy

        # A PPO model built for Pendulum-v1 (obs shape (3,)) must be
        # rejected against InvertedPendulum-v5 (obs shape (4,)).
        other = gym.make("Pendulum-v1")
        model = PPO("MlpPolicy", other, n_steps=32, verbose=0)
        with pytest.raises(SpaceMismatchError, match="trained on observations"):
            validate_spaces(SB3Policy(model), env)
        other.close()

    def test_missing_file(self, env):
        with pytest.raises(FileNotFoundError):
            load_policy("nonexistent.zip", env)

    def test_zip_requires_algo(self, env, tmp_path):
        p = tmp_path / "model.zip"
        p.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="algorithm class"):
            load_policy(p, env)


class TestRunner:
    def test_rollout_records_structure(self, env):
        policy = load_policy(ZeroPolicy(), env)
        record = rollout_one(policy, env, seed=42, success_fn=survival_success)
        assert record.seed == 42
        assert record.length >= 1
        # v5 pays +1 per step except the terminating step, which pays 0
        assert record.total_reward == pytest.approx(record.length - int(record.terminated))
        assert len(record.step_infos) == record.length + 1  # includes reset info
        assert JOINT_VELOCITIES in record.step_infos[-1]
        assert record.terminated or record.truncated

    def test_reproducible_across_runs(self, env):
        policy = load_policy(ZeroPolicy(), env)
        a = run_rollouts(policy, env, 3, survival_success, base_seed=7)
        b = run_rollouts(policy, env, 3, survival_success, base_seed=7)
        assert a.seeds == b.seeds
        assert a.lengths == b.lengths
        assert a.returns == b.returns

    def test_success_fn_wiring(self, env):
        policy = load_policy(ZeroPolicy(), env)
        never = run_rollouts(policy, env, 2, lambda *a: False, base_seed=1)
        always = run_rollouts(policy, env, 2, lambda *a: True, base_seed=1)
        assert never.successes == [False, False]
        assert always.successes == [True, True]
        assert always.success_rate == 1.0

    def test_max_steps_truncates(self, env):
        policy = load_policy(ZeroPolicy(), env)
        record = rollout_one(policy, env, seed=3, success_fn=survival_success, max_steps=5)
        assert record.length <= 5

    def test_seed_sequence_deterministic(self):
        assert make_seed_sequence(5, 0) == make_seed_sequence(5, 0)
        assert make_seed_sequence(5, 0) != make_seed_sequence(5, 1)

    def test_too_few_seeds_rejected(self, env):
        policy = load_policy(ZeroPolicy(), env)
        with pytest.raises(ValueError, match="seeds"):
            run_rollouts(policy, env, 5, survival_success, seeds=[1, 2])
