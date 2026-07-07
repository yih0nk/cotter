"""Dict-observation-space support: policy validation, wrapper, runner.

Uses FetchReachDense-v4 (gymnasium-robotics): Dict obs with
observation/achieved_goal/desired_goal keys, MuJoCo-backed, is_success
in the step info dict.
"""

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO

from cotter.envs.registry import make_env_by_id
from cotter.envs.wrapper import INSTRUMENTED_KEYS, JOINT_VELOCITIES, CotterWrapper
from cotter.policy import SB3Policy, SpaceMismatchError, load_policy, validate_spaces
from cotter.runner import run_rollouts
from cotter.success import make_success_fn

FETCH = "FetchReachDense-v4"


@pytest.fixture(scope="module")
def fetch_env():
    env = CotterWrapper(make_env_by_id(FETCH))
    yield env
    env.close()


@pytest.fixture(scope="module")
def fetch_model(fetch_env):
    # untrained model: exercises spaces and prediction, not task skill
    return PPO("MultiInputPolicy", fetch_env, n_steps=32, verbose=0, device="cpu")


class TestRegistry:
    def test_fetch_env_resolves(self):
        env = make_env_by_id(FETCH)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        env.close()

    def test_unknown_env_still_raises(self):
        from gymnasium.error import NameNotFound

        with pytest.raises(NameNotFound):
            make_env_by_id("NotARealEnv-v0")


class TestDictSpaceValidation:
    def test_matching_dict_policy_validates(self, fetch_env, fetch_model):
        policy = load_policy(fetch_model, fetch_env, name="fetch")
        obs, _ = fetch_env.reset(seed=0)
        action = policy.predict(obs)
        assert action.shape == fetch_env.action_space.shape

    def test_box_policy_on_dict_env_fails_loudly(self, fetch_env):
        pendulum = gym.make("InvertedPendulum-v5")
        model = PPO("MlpPolicy", pendulum, n_steps=32, verbose=0, device="cpu")
        with pytest.raises(SpaceMismatchError, match="trained on observations"):
            validate_spaces(SB3Policy(model), fetch_env)
        pendulum.close()

    def test_dict_policy_on_box_env_fails_loudly(self, fetch_model):
        pendulum = gym.make("InvertedPendulum-v5")
        with pytest.raises(SpaceMismatchError, match="trained on observations"):
            validate_spaces(SB3Policy(fetch_model), pendulum)
        pendulum.close()

    def test_dict_key_shape_mismatch_fails_loudly(self, fetch_model):
        # FetchPush has the same keys but a bigger 'observation' vector, so a
        # Reach policy must be rejected per-key, not just per-key-set.
        push = make_env_by_id("FetchPushDense-v4")
        with pytest.raises(SpaceMismatchError, match="trained on observations"):
            validate_spaces(SB3Policy(fetch_model), push)
        push.close()


class TestWrapperAndRunnerWithDictObs:
    def test_wrapper_instruments_fetch(self, fetch_env):
        _, info = fetch_env.reset(seed=0)
        for key in INSTRUMENTED_KEYS:
            assert key in info
        assert info[JOINT_VELOCITIES].shape == (15,)  # fetch robot DoFs

    def test_rollouts_with_info_flag_success(self, fetch_env, fetch_model):
        policy = load_policy(fetch_model, fetch_env)
        success_fn = make_success_fn({"type": "info_flag", "key": "is_success"})
        rollouts = run_rollouts(policy, fetch_env, 2, success_fn, base_seed=0)
        assert len(rollouts.records) == 2
        assert all(r.length == 50 for r in rollouts.records)  # fixed horizon
        assert all(isinstance(s, bool) for s in rollouts.successes)
        # instrumented infos recorded every step for safety checks
        assert all(JOINT_VELOCITIES in i for ep in rollouts.episode_infos for i in ep)
