"""Tests for the simulator backend abstraction.

No real Isaac Sim is needed or expected: the isaac-sim backend is
verified to fail fast on a machine without omni.isaac.gym.
"""

import gymnasium as gym
import pytest

from cotter.backends import (
    BackendFactory,
    BackendNotAvailableError,
    GymnasiumBackend,
    IsaacSimBackend,
)
from cotter.envs.wrapper import CotterWrapper


class TestRegistry:
    def test_gymnasium_registered_and_default_available(self):
        assert "gymnasium" in BackendFactory.available()
        assert "isaac-sim" in BackendFactory.available()

    def test_from_name_returns_gymnasium(self):
        backend = BackendFactory.from_name("gymnasium")
        assert isinstance(backend, GymnasiumBackend)
        assert backend.name() == "gymnasium"

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown backend 'nope'"):
            BackendFactory.from_name("nope")


class TestGymnasiumBackend:
    def test_wraps_mujoco_env(self):
        env = GymnasiumBackend().make_env("InvertedPendulum-v5")
        assert isinstance(env, CotterWrapper)
        env.close()

    def test_non_mujoco_env_passthrough(self):
        env = GymnasiumBackend().make_env("CartPole-v1")
        assert not isinstance(env, CotterWrapper)
        assert isinstance(env, gym.Env)
        env.close()


class TestIsaacSimBackend:
    def test_construction_fails_without_isaac(self):
        # omni.isaac.gym is not installed on the CPU dev/CI machine.
        with pytest.raises(BackendNotAvailableError, match="omni.isaac.gym"):
            IsaacSimBackend()

    def test_from_name_surfaces_unavailability(self):
        with pytest.raises(BackendNotAvailableError):
            BackendFactory.from_name("isaac-sim")


class TestCustomBackendMocking:
    def test_register_and_use_a_fake_backend(self):
        # Backends self-register via __init_subclass__; a test double can
        # stand in for a real simulator without any heavy dependency.
        class FakeBackend(BackendFactory):
            backend_name = "fake-test-backend"

            def make_env(self, env_id: str):
                return gym.make("CartPole-v1")

        try:
            backend = BackendFactory.from_name("fake-test-backend")
            assert backend.name() == "fake-test-backend"
            env = backend.make_env("anything")
            assert isinstance(env, gym.Env)
            env.close()
        finally:
            BackendFactory._registry.pop("fake-test-backend", None)
