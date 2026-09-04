"""Tests for the ISO/TS 15066 limits library and PFL physics."""

import math

import pytest

from cotter.tests.iso15066 import (
    ISO_TS_15066_LIMITS,
    body_region,
    collision_energy,
    max_relative_speed,
    peak_force,
    reduced_mass,
)


class TestLimitsLibrary:
    def test_regions_present(self):
        for name in ("chest", "hand_finger", "skull_forehead", "face"):
            assert name in ISO_TS_15066_LIMITS

    def test_transient_defaults_to_twice_quasi_static(self):
        r = body_region("x", quasi_static_force=100, max_pressure=30, spring_constant=25, human_mass=40)
        assert r.transient_force == 200

    def test_explicit_transient_kept(self):
        r = body_region("x", 100, 30, 25, 40, transient_force=150)
        assert r.transient_force == 150

    def test_chest_values(self):
        chest = ISO_TS_15066_LIMITS["chest"]
        assert chest.quasi_static_force == 140
        assert chest.transient_force == 280


class TestPhysics:
    def test_reduced_mass_formula(self):
        assert reduced_mass(6.0, 3.0) == pytest.approx(2.0)  # 1/(1/6+1/3)=2

    def test_reduced_mass_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            reduced_mass(0.0, 3.0)

    def test_collision_energy(self):
        assert collision_energy(2.0, 3.0) == pytest.approx(9.0)  # 0.5*2*9

    def test_peak_force_and_max_speed_are_inverse(self):
        mu = reduced_mass(5.0, 40.0)
        k = 25  # N/mm
        f_limit = 140.0
        v_max = max_relative_speed(f_limit, k, mu)
        # a collision at v_max produces exactly the force limit
        assert peak_force(k, mu, v_max) == pytest.approx(f_limit)

    def test_peak_force_units(self):
        # F = v * sqrt(k[N/m] * mu); k=25 N/mm = 25000 N/m
        mu = reduced_mass(5.0, 40.0)
        assert peak_force(25, mu, 1.0) == pytest.approx(math.sqrt(25000 * mu))

    def test_stiffer_region_has_lower_speed_limit(self):
        mu = reduced_mass(5.0, 4.4)
        soft = max_relative_speed(140, 25, mu)   # chest-like: soft
        stiff = max_relative_speed(140, 150, mu)  # skull-like: stiff
        assert stiff < soft  # a stiffer contact reaches the force limit at lower speed
