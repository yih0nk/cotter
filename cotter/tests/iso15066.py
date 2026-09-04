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
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


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


_CONTACT_TYPES = ("transient", "quasi_static")


class PFLDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class RegionLimit:
    """The speed/force limit computed for one region against a given robot mass."""

    name: str
    force_limit: float  # N (transient or quasi-static, per contact_type)
    speed_limit: float  # m/s, the max permissible relative contact speed
    human_mass: float  # kg

    def to_dict(self) -> dict:
        return {
            "region": self.name,
            "force_limit": self.force_limit,
            "speed_limit": self.speed_limit,
            "human_mass": self.human_mass,
        }


@dataclass(frozen=True)
class PFLViolation:
    trial: int
    timestep: int
    region: str
    speed: float
    speed_limit: float
    implied_force: float
    force_limit: float

    def to_dict(self) -> dict:
        return {
            "trial": self.trial,
            "timestep": self.timestep,
            "region": self.region,
            "speed": self.speed,
            "speed_limit": self.speed_limit,
            "implied_force": self.implied_force,
            "force_limit": self.force_limit,
        }


@dataclass
class PFLResult:
    decision: PFLDecision
    m_robot: float
    contact_type: str
    speed_key: str
    region_limits: list[RegionLimit]
    n_trials: int
    n_timesteps_checked: int
    worst_speed: float = 0.0
    binding_region: str = ""  # the most restrictive region (lowest speed limit)
    violations: list[PFLViolation] = field(default_factory=list)

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "m_robot": self.m_robot,
            "contact_type": self.contact_type,
            "speed_key": self.speed_key,
            "binding_region": self.binding_region,
            "worst_speed": self.worst_speed,
            "region_limits": [r.to_dict() for r in self.region_limits],
            "n_trials": self.n_trials,
            "n_timesteps_checked": self.n_timesteps_checked,
            "n_violations": self.n_violations,
            "violations": [v.to_dict() for v in self.violations[:50]],
        }


def region_limits(
    regions: Sequence[BodyRegion], m_robot: float, contact_type: str = "transient"
) -> list[RegionLimit]:
    """Compute the permissible contact-speed limit for each region."""
    if contact_type not in _CONTACT_TYPES:
        raise ValueError(f"contact_type must be one of {list(_CONTACT_TYPES)}; got {contact_type!r}")
    out: list[RegionLimit] = []
    for r in regions:
        mu = reduced_mass(m_robot, r.human_mass)
        force_limit = r.transient_force if contact_type == "transient" else r.quasi_static_force
        out.append(
            RegionLimit(
                name=r.name,
                force_limit=force_limit,
                speed_limit=max_relative_speed(force_limit, r.spring_constant, mu),
                human_mass=r.human_mass,
            )
        )
    return out


def evaluate_pfl(
    episode_infos: Sequence[Sequence[Mapping]],
    regions: Sequence[BodyRegion],
    m_robot: float,
    speed_key: str,
    contact_type: str = "transient",
) -> PFLResult:
    """Check TCP speed against ISO/TS 15066 per-region limits over rollouts.

    ``speed_key`` names the per-step info entry holding the robot's TCP
    linear velocity (scalar speed or a velocity vector — its L2 norm is
    used). For each region, a collision at the TCP speed would exceed the
    region's force limit iff the speed exceeds that region's permissible
    contact speed; any such timestep fails the check. Raises ``KeyError``
    if ``speed_key`` is absent (a misconfiguration must not silently pass).
    """
    if not regions:
        raise ValueError("evaluate_pfl called with no regions configured")
    limits = region_limits(regions, m_robot, contact_type)
    # cache mu per region for the implied-force computation
    mus = {r.name: reduced_mass(m_robot, r.human_mass) for r in regions}
    ks = {r.name: r.spring_constant for r in regions}
    binding = min(limits, key=lambda rl: rl.speed_limit)

    violations: list[PFLViolation] = []
    worst_speed = 0.0
    n_steps = 0

    for trial, steps in enumerate(episode_infos):
        for timestep, info in enumerate(steps):
            if speed_key not in info:
                raise KeyError(
                    f"ISO/TS 15066 check references speed_key '{speed_key}' but the step "
                    f"info only contains {sorted(info.keys())}. Expose the TCP linear "
                    "velocity under this key."
                )
            n_steps += 1
            speed = float(np.linalg.norm(np.atleast_1d(np.asarray(info[speed_key], dtype=float))))
            worst_speed = max(worst_speed, speed)
            for rl in limits:
                if speed > rl.speed_limit:
                    violations.append(
                        PFLViolation(
                            trial=trial,
                            timestep=timestep,
                            region=rl.name,
                            speed=speed,
                            speed_limit=rl.speed_limit,
                            implied_force=peak_force(ks[rl.name], mus[rl.name], speed),
                            force_limit=rl.force_limit,
                        )
                    )

    return PFLResult(
        decision=PFLDecision.FAIL if violations else PFLDecision.PASS,
        m_robot=m_robot,
        contact_type=contact_type,
        speed_key=speed_key,
        region_limits=limits,
        n_trials=len(episode_infos),
        n_timesteps_checked=n_steps,
        worst_speed=worst_speed,
        binding_region=binding.name,
        violations=violations,
    )
