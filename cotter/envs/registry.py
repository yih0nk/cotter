"""Environment creation with optional extension registration.

gymnasium-robotics environments (Fetch, Shadow Hand, ...) are not
registered until the package is imported. :func:`make_env_by_id` retries
a failed ``gym.make`` after registering installed extension packages, so
config files can name any installed env without boilerplate.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.error import NameNotFound

_registered = False


def register_extension_envs() -> None:
    """Register env packages that require an explicit import (idempotent)."""
    global _registered
    if _registered:
        return
    try:
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
    except ImportError:
        pass
    _registered = True


def make_env_by_id(env_id: str, **kwargs) -> gym.Env:
    """``gym.make`` that knows about installed extension packages."""
    try:
        return gym.make(env_id, **kwargs)
    except NameNotFound:
        register_extension_envs()
        return gym.make(env_id, **kwargs)
