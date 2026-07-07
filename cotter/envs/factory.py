"""Picklable environment factory for parallel rollouts.

``AsyncVectorEnv`` constructs its sub-environments inside worker
processes, so it needs a factory that survives pickling under the
``spawn`` start method used on macOS. A module-level callable class does;
a local closure does not. :class:`WrappedEnvFactory` rebuilds the same
instrumented env that :func:`cotter.pipeline.make_env` produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym

from cotter.envs.registry import make_env_by_id
from cotter.envs.wrapper import CotterWrapper


@dataclass(frozen=True)
class WrappedEnvFactory:
    """Zero-argument callable that builds one wrapped env by id."""

    env_id: str

    def __call__(self) -> gym.Env:
        env = make_env_by_id(self.env_id)
        try:
            return CotterWrapper(env)
        except TypeError:
            return env
