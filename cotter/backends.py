"""Simulator backend abstraction.

A backend knows how to turn an env id into a ready-to-test Gymnasium
environment (instrumented with :class:`~cotter.envs.wrapper.CotterWrapper`
when possible). This isolates the rest of the framework from *which*
simulator is behind the Gymnasium API, so the same test battery can run
against MuJoCo today and other Gymnasium-compatible simulators later.

Backends are looked up by name via :meth:`BackendFactory.from_name`;
``RunConfig.backend`` selects one (default ``"gymnasium"``). A backend
that needs an optional dependency raises :class:`BackendNotAvailableError`
at construction, so an unavailable backend fails immediately with a clear
message rather than deep inside a run.
"""

from __future__ import annotations

import abc

import gymnasium as gym

from cotter.envs.registry import make_env_by_id
from cotter.envs.wrapper import CotterWrapper


class BackendNotAvailableError(RuntimeError):
    """A backend's required simulator/dependency is not importable."""


class BackendFactory(abc.ABC):
    """Interface for simulator backends.

    Concrete backends set a class-level ``backend_name`` (used for
    registration and lookup) and implement :meth:`make_env`.
    """

    backend_name: str = ""
    _registry: dict[str, type["BackendFactory"]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.backend_name:
            BackendFactory._registry[cls.backend_name] = cls

    @abc.abstractmethod
    def make_env(self, env_id: str) -> gym.Env:
        """Create an instrumented environment for ``env_id``."""

    def name(self) -> str:
        return self.backend_name

    @classmethod
    def from_name(cls, name: str) -> "BackendFactory":
        """Construct the backend registered under ``name``.

        Raises ``ValueError`` for an unknown name and
        :class:`BackendNotAvailableError` if the named backend's
        dependencies are missing (surfaced from its constructor).
        """
        if name not in cls._registry:
            raise ValueError(
                f"unknown backend '{name}' (available: {sorted(cls._registry)})"
            )
        return cls._registry[name]()

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)


class GymnasiumBackend(BackendFactory):
    """Default backend: standard Gymnasium/MuJoCo via ``gym.make``."""

    backend_name = "gymnasium"

    def make_env(self, env_id: str) -> gym.Env:
        env = make_env_by_id(env_id)
        try:
            return CotterWrapper(env)
        except TypeError:
            # not MuJoCo-backed: safety instrumentation is unavailable, but
            # performance/regression/adversarial still work on the raw env
            return env


class IsaacSimBackend(BackendFactory):
    """NVIDIA Isaac Sim backend (optional, GPU/cluster only).

    Isaac Sim exposes a Gymnasium-compatible API but requires the
    ``omni.isaac.gym`` package, which is not importable on CPU-only
    machines. Construction fails fast when it is absent; when present,
    envs are built through Isaac's Gymnasium interface and instrumented
    with :class:`CotterWrapper` like any other MuJoCo/Gymnasium env.

    This path is not exercised on the CPU development machine; it is
    validated on a GPU cluster separately.
    """

    backend_name = "isaac-sim"

    def __init__(self) -> None:
        import importlib.util

        # find_spec returns None for a missing leaf, but raises
        # ModuleNotFoundError when an intermediate package (omni) is absent;
        # both mean the backend is unavailable.
        try:
            spec = importlib.util.find_spec("omni.isaac.gym")
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            raise BackendNotAvailableError(
                "the isaac-sim backend requires the 'omni.isaac.gym' package "
                "from NVIDIA Isaac Sim, which is not installed. Install Isaac "
                "Sim on a CUDA machine, or use the default 'gymnasium' backend."
            )

    def make_env(self, env_id: str) -> gym.Env:  # pragma: no cover - no Isaac on CI
        # Isaac registers Gymnasium-compatible envs; build and instrument
        # them through the same path as any MuJoCo env.
        env = make_env_by_id(env_id)
        try:
            return CotterWrapper(env)
        except TypeError:
            return env
