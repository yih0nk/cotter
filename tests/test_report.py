"""Unit tests for the TestReport container."""

import json

import numpy as np
import pytest

from cotter.report import TestReport
from cotter.tests.regression import mcnemar_exact
from cotter.tests.safety import SafetyLimit, evaluate_safety
from cotter.tests.sprt import run_sprt


@pytest.fixture
def report():
    return TestReport(policy_name="demo", env_id="InvertedPendulum-v5", metadata={"seed": 0})


def make_sprt_pass():
    return run_sprt(iter([True] * 10), p0=0.5, p1=0.9)


def make_safety_fail():
    limits = [SafetyLimit("v", 1.0)]
    return evaluate_safety([[{"v": np.array([2.0])}]], limits)


def make_no_regression():
    return mcnemar_exact([True, True, False], [True, True, False])


class TestAdapters:
    def test_sprt_pass_maps_to_passed(self, report):
        report.add_sprt(make_sprt_pass())
        r = report.results[0]
        assert (r.category, r.passed) == ("performance", True)
        assert "PASS" in r.summary

    def test_sprt_inconclusive_maps_to_none(self, report):
        res = run_sprt(iter([True, False] * 5), p0=0.4, p1=0.6, n_max=10)
        report.add_sprt(res)
        assert report.results[0].passed is None
        assert report.overall_passed  # informational does not fail the report

    def test_safety_fail_maps_and_summarizes_violation(self, report):
        report.add_safety(make_safety_fail())
        r = report.results[0]
        assert r.passed is False
        assert "trial 0 step 0" in r.summary
        assert not report.overall_passed

    def test_regression_adapter(self, report):
        report.add_regression(make_no_regression())
        r = report.results[0]
        assert r.category == "regression"
        assert r.passed is True
        assert "p=1" in r.summary


class TestOutput:
    def test_overall_pass_logic(self, report):
        report.add_sprt(make_sprt_pass())
        assert report.overall_passed
        report.add_safety(make_safety_fail())
        assert not report.overall_passed

    def test_json_roundtrip_with_numpy_values(self, report, tmp_path):
        report.add_safety(make_safety_fail())
        report.add("adversarial", "random", True, "ok", {"rate": np.float64(0.5), "n": np.int64(3)})
        path = report.to_json(tmp_path / "out" / "report.json")
        loaded = json.loads(path.read_text())
        assert loaded["policy_name"] == "demo"
        assert loaded["overall_passed"] is False
        assert loaded["results"][1]["data"]["rate"] == 0.5
        assert len(loaded["results"]) == 2

    def test_summary_lists_all_categories(self, report):
        report.add_sprt(make_sprt_pass())
        report.add_safety(make_safety_fail())
        report.add_regression(make_no_regression())
        text = report.summary()
        assert "COTTER TEST REPORT" in text
        for token in ("performance/", "safety/", "regression/", "OVERALL: FAIL"):
            assert token in text
