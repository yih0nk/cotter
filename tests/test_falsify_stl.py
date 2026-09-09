"""Integration tests for the STL-robustness falsification objective."""

import gymnasium as gym
import pytest
import torch

pytest.importorskip("cma")
pytest.importorskip("rtamt")

from cotter.envs.wrapper import CotterWrapper
from cotter.falsify import falsify, stl_scenario_objective
from cotter.policy import load_policy

ENV_ID = "InvertedPendulum-v5"
VARS = {"jv0": {"key": "cotter/joint_velocities", "index": 0}}


class ConstPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def env_factory():
    return CotterWrapper(gym.make(ENV_ID))


@pytest.fixture
def policy():
    return load_policy(ConstPolicy(), env_factory())


def _noop(env, params):
    pass


class TestStlScenarioObjective:
    def test_objective_returns_robustness(self, policy):
        obj = stl_scenario_objective(
            policy, env_factory, _noop, "always (jv0 <= 100)", VARS, n_episodes=1
        )
        value = obj([0.0])
        # a satisfied spec -> positive robustness
        assert value > 0

    def test_objective_is_deterministic(self, policy):
        obj = stl_scenario_objective(
            policy, env_factory, _noop, "always (jv0 <= 100)", VARS, base_seed=7
        )
        assert obj([0.0]) == pytest.approx(obj([0.0]))

    def test_falsify_reports_violation_for_impossible_spec(self, policy):
        # "jv0 <= -100" can never hold -> robustness always negative -> falsified fast
        obj = stl_scenario_objective(
            policy, env_factory, _noop, "always (jv0 <= -100)", VARS
        )
        res = falsify(obj, lower=[-1.0], upper=[1.0], max_evaluations=20, seed=0)
        assert res.falsified
        assert res.best_value < 0

    def test_falsify_finds_no_counterexample_for_satisfied_spec(self, policy):
        obj = stl_scenario_objective(policy, env_factory, _noop, "always (jv0 <= 100)", VARS)
        res = falsify(obj, lower=[-1.0], upper=[1.0], max_evaluations=15, seed=0)
        assert not res.falsified
