"""Tests for STL specification evaluation (cotter.tests.stl)."""

import pytest

pytest.importorskip("rtamt")

from cotter.tests.stl import (
    evaluate_stl,
    extract_dataset,
    episode_robustness,
)

VARS = {"speed": {"key": "tcp_speed"}}


def _episode(speeds, key="tcp_speed"):
    return [{key: s} for s in speeds]


class TestExtractDataset:
    def test_scalar_signal(self):
        ds = extract_dataset(_episode([0.1, 0.2]), VARS)
        assert ds["time"] == [0, 1]
        assert ds["speed"] == [0.1, 0.2]

    def test_indexed_component(self):
        steps = [{"vel": [1.0, 9.0]}, {"vel": [2.0, 8.0]}]
        ds = extract_dataset(steps, {"vx": {"key": "vel", "index": 0}})
        assert ds["vx"] == [1.0, 2.0]

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            extract_dataset([{"other": 1.0}], VARS)

    def test_nonscalar_without_index_raises(self):
        with pytest.raises(ValueError):
            extract_dataset([{"tcp_speed": [1.0, 2.0]}], VARS)


class TestEvaluateStl:
    def test_satisfied_spec_positive_robustness(self):
        r = evaluate_stl([_episode([0.5, 0.8, 0.4])], "always (speed <= 1.5)", VARS)
        assert r.passed
        assert r.min_robustness > 0
        assert r.min_robustness == pytest.approx(1.5 - 0.8)

    def test_violated_spec_negative_robustness(self):
        r = evaluate_stl([_episode([0.5, 2.0, 0.4])], "always (speed <= 1.5)", VARS)
        assert not r.passed
        assert r.min_robustness == pytest.approx(1.5 - 2.0)

    def test_worst_episode_is_reported(self):
        eps = [_episode([0.1, 0.2]), _episode([0.1, 3.0]), _episode([0.5])]
        r = evaluate_stl(eps, "always (speed <= 1.5)", VARS)
        assert r.worst_episode == 1
        assert r.n_episodes == 3
        assert len(r.per_episode) == 3

    def test_threshold_shifts_pass(self):
        # min robustness ~0.7; a threshold above it should fail
        eps = [_episode([0.5, 0.8])]
        assert evaluate_stl(eps, "always (speed <= 1.5)", VARS, threshold=0.0).passed
        assert not evaluate_stl(eps, "always (speed <= 1.5)", VARS, threshold=1.0).passed

    def test_eventually_operator(self):
        # speed eventually drops to <= 0.1
        r = evaluate_stl([_episode([1.0, 0.5, 0.05])], "eventually (speed <= 0.1)", VARS)
        assert r.passed

    def test_bad_formula_raises(self):
        with pytest.raises(ValueError):
            evaluate_stl([_episode([0.1])], "always (speed <<< 1.5)", VARS)

    def test_no_episodes_raises(self):
        with pytest.raises(ValueError):
            evaluate_stl([], "always (speed <= 1.5)", VARS)

    def test_summary_and_dict(self):
        r = evaluate_stl([_episode([0.5])], "always (speed <= 1.5)", VARS, name="cap")
        assert "STL 'cap'" in r.summary()
        assert r.to_dict()["formula"] == "always (speed <= 1.5)"

    def test_episode_robustness_helper(self):
        rob = episode_robustness(_episode([0.5, 2.0]), "always (speed <= 1.5)", VARS)
        assert rob == pytest.approx(-0.5)
