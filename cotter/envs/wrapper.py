"""Gymnasium wrapper that surfaces safety-relevant MuJoCo quantities.

:class:`CotterWrapper` guarantees that every ``step()`` info dict contains
the physical quantities the safety tests read, copied out of the MuJoCo
``data`` structure of the wrapped environment:

==============================  =========================================
info key                        contents
==============================  =========================================
``cotter/joint_velocities``      ``data.qvel`` — generalized joint
                                 velocities (rad/s or m/s per DoF)
``cotter/actuator_forces``       ``data.actuator_force`` — scalar force/
                                 torque produced by each actuator (N or Nm)
``cotter/contact_count``         ``data.ncon`` — number of active contact
                                 points this step
==============================  =========================================

The wrapper works with any Gymnasium MuJoCo environment (anything whose
unwrapped env exposes a ``data`` attribute with these fields) and fails
loudly at construction time otherwise.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

JOINT_VELOCITIES = "cotter/joint_velocities"
ACTUATOR_FORCES = "cotter/actuator_forces"
CONTACT_COUNT = "cotter/contact_count"

INSTRUMENTED_KEYS = (JOINT_VELOCITIES, ACTUATOR_FORCES, CONTACT_COUNT)


class CotterWrapper(gym.Wrapper):
    """Instrument a MuJoCo env's info dict with safety-relevant quantities."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        data = getattr(env.unwrapped, "data", None)
        if data is None or not all(
            hasattr(data, attr) for attr in ("qvel", "actuator_force", "ncon")
        ):
            raise TypeError(
                f"CotterWrapper requires a MuJoCo-backed environment exposing "
                f"`unwrapped.data` with qvel/actuator_force/ncon; "
                f"got {type(env.unwrapped).__name__}. Wrap a Gymnasium MuJoCo "
                "env (e.g. gym.make('InvertedPendulum-v5'))."
            )

    def _instrument(self, info: dict) -> dict:
        data = self.env.unwrapped.data
        info[JOINT_VELOCITIES] = np.array(data.qvel, dtype=float, copy=True)
        info[ACTUATOR_FORCES] = np.array(data.actuator_force, dtype=float, copy=True)
        info[CONTACT_COUNT] = int(data.ncon)
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._instrument(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._instrument(info)
