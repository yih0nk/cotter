"""Integration tests for the ONNX policy loader.

These load a committed 4->1 ONNX fixture (weights are irrelevant — the
point is the load/predict/validate path) against a real InvertedPendulum
env, so they exercise onnxruntime for real. Skipped if onnxruntime is
not installed.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from cotter.envs.wrapper import CotterWrapper
from cotter.policy import OnnxPolicy, SpaceMismatchError, load_policy
from cotter.runner import rollout_one

ONNX_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "probe_onnx_inverted_pendulum.onnx"
)
ENV_ID = "InvertedPendulum-v5"


@pytest.fixture
def env():
    e = CotterWrapper(gym.make(ENV_ID))
    yield e
    e.close()


def survival_success(total_reward, length, terminated, truncated, final_info):
    return length >= 20


class TestOnnxLoading:
    def test_load_policy_returns_onnx_policy(self, env):
        policy = load_policy(ONNX_FIXTURE, env)
        assert isinstance(policy, OnnxPolicy)
        assert policy.name == "probe_onnx_inverted_pendulum"

    def test_name_override(self, env):
        policy = load_policy(ONNX_FIXTURE, env, name="grasp_v3")
        assert policy.name == "grasp_v3"

    def test_predict_shape_and_finiteness(self, env):
        policy = load_policy(ONNX_FIXTURE, env)
        obs, _ = env.reset(seed=0)
        action = policy.predict(obs)
        assert action.shape == env.action_space.shape
        assert np.all(np.isfinite(action))

    def test_batch_axis_is_added_and_stripped(self, env):
        # the fixture declares a batched input; a single (4,) obs must
        # round-trip to a single (1,) action, not a batched (1, 1).
        policy = load_policy(ONNX_FIXTURE, env)
        obs = env.observation_space.sample()
        assert obs.ndim == 1
        assert policy.predict(obs).shape == (1,)


class TestOnnxIntegration:
    def test_rollout_runs_with_onnx_policy(self, env):
        policy = load_policy(ONNX_FIXTURE, env)
        record = rollout_one(policy, env, seed=42, success_fn=survival_success)
        assert record.length >= 1
        assert record.terminated or record.truncated


class TestOnnxErrors:
    def test_missing_file(self, env):
        with pytest.raises(FileNotFoundError):
            load_policy("nope.onnx", env)

    def test_wrong_env_shape_is_rejected(self):
        # a Box(7) env does not match the 4-dim ONNX input -> the functional
        # probe in validate_spaces must fail loudly.
        mismatched = gym.make("Reacher-v5")  # obs dim 10, not 4
        try:
            with pytest.raises(SpaceMismatchError):
                load_policy(ONNX_FIXTURE, mismatched)
        finally:
            mismatched.close()
