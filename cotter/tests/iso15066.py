"""ISO/TS 15066 power-and-force-limiting (PFL) checks.

ISO/TS 15066 specifies, per human body region, the maximum permissible
contact force / pressure for a collaborative robot, together with an
effective spring constant ``k`` and the biomechanical collision model.
From these, a maximum permissible relative contact speed can be
back-solved and checked per timestep against the robot's TCP speed — the
single most directly checkable numeric requirement in the collaborative
robotics standards, and pure CPU arithmetic over rollout data.

Collision model (ISO/TS 15066 Annex A):

    reduced mass         1/mu = 1/m_robot + 1/m_human
    transferred energy   E    = 0.5 * mu * v_rel^2
    peak contact force   F    = v_rel * sqrt(k * mu)
    => max relative speed v_max = F_limit / sqrt(k * mu)

Transient (short, dynamic impact) force limits are ~2x the quasi-static
(clamping) limits — a different injury mechanism.

.. warning::
   The exact ISO/TS 15066 Annex A tables are paywalled. The values in
   :data:`ISO_TS_15066_LIMITS` are the widely-published approximations
   and are provided as **editable defaults** — confirm them against the
   purchased standard before relying on them for certification, and note
   ISO/TS 15066 is under revision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BodyRegion:
    """ISO/TS 15066 biomechanical limits for one human body region.

    ``spring_constant`` is in N/mm (as the standard tabulates it);
    ``human_mass`` is the region's effective mass in kg. ``transient_force``
    defaults to twice the quasi-static value when constructed via
    :func:`body_region`.
    """

    name: str
    quasi_static_force: float  # N
    transient_force: float  # N (~2x quasi-static)
    max_pressure: float  # N/cm^2 (reference; pressure needs a contact area to check)
    spring_constant: float  # N/mm
    human_mass: float  # kg, effective mass of the region


def body_region(
    name: str,
    quasi_static_force: float,
    max_pressure: float,
    spring_constant: float,
    human_mass: float,
    transient_force: float | None = None,
) -> BodyRegion:
    """Build a :class:`BodyRegion`, defaulting transient force to 2x quasi-static."""
    return BodyRegion(
        name=name,
        quasi_static_force=quasi_static_force,
        transient_force=transient_force if transient_force is not None else 2.0 * quasi_static_force,
        max_pressure=max_pressure,
        spring_constant=spring_constant,
        human_mass=human_mass,
    )


# Widely-published ISO/TS 15066 approximations — EDITABLE DEFAULTS, verify
# against the purchased standard (see module warning).
ISO_TS_15066_LIMITS: dict[str, BodyRegion] = {
    r.name: r
    for r in (
        body_region("skull_forehead", 130, 30, 150, 4.4),
        body_region("face", 65, 20, 75, 4.4),
        body_region("neck", 150, 50, 50, 1.2),
        body_region("chest", 140, 45, 25, 40.0),
        body_region("abdomen", 110, 20, 10, 40.0),
        body_region("pelvis", 180, 50, 25, 40.0),
        body_region("upper_arm_shoulder", 150, 50, 30, 3.0),
        body_region("forearm_wrist", 160, 60, 40, 2.0),
        body_region("hand_finger", 140, 75, 75, 0.6),
        body_region("thigh_knee", 220, 50, 50, 8.0),
        body_region("lower_leg", 130, 60, 60, 4.0),
    )
}


def reduced_mass(m_robot: float, m_human: float) -> float:
    """Two-body reduced mass ``1/mu = 1/m_robot + 1/m_human`` (kg)."""
    if m_robot <= 0 or m_human <= 0:
        raise ValueError("masses must be positive")
    return 1.0 / (1.0 / m_robot + 1.0 / m_human)


def collision_energy(mu: float, v_rel: float) -> float:
    """Transferred collision energy ``E = 0.5 * mu * v_rel^2`` (J)."""
    return 0.5 * mu * v_rel * v_rel


def peak_force(spring_constant_n_per_mm: float, mu: float, v_rel: float) -> float:
    """Peak contact force ``F = v_rel * sqrt(k * mu)`` (N).

    ``k`` is given in N/mm and converted to N/m for SI consistency.
    """
    k = spring_constant_n_per_mm * 1000.0  # N/mm -> N/m
    return v_rel * math.sqrt(k * mu)


def max_relative_speed(force_limit: float, spring_constant_n_per_mm: float, mu: float) -> float:
    """Max relative contact speed keeping the peak force within ``force_limit`` (m/s)."""
    k = spring_constant_n_per_mm * 1000.0
    return force_limit / math.sqrt(k * mu)
