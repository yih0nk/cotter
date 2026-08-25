"""Unit tests for Markdown report rendering (cotter.render_md)."""

import numpy as np
import pytest

from cotter.render_md import render_markdown
from cotter.report import TestReport
from cotter.tests.safety import SafetyLimit, evaluate_safety
from cotter.tests.sprt import run_sprt


@pytest.fixture
def report():
    r = TestReport(policy_name="victim", env_id="InvertedPendulum-v5")
    r.add_sprt(run_sprt(iter([True] * 10), p0=0.5, p1=0.9))  # pass
    r.add_safety(evaluate_safety([[{"v": np.array([2.0])}]], [SafetyLimit("v", 1.0)]))  # fail
    return r


class TestRenderMarkdown:
    def test_header_and_verdict(self, report):
        md = render_markdown(report.to_dict())
        assert md.startswith("# Cotter test report")
        assert "victim" in md
        assert "InvertedPendulum-v5" in md
        assert "OVERALL: ❌ **FAIL**" in md  # a safety violation was added

    def test_counts(self, report):
        md = render_markdown(report.to_dict())
        assert "`1` passing" in md
        assert "`1` failing" in md
        assert "`0` informational" in md

    def test_one_row_per_result(self, report):
        md = render_markdown(report.to_dict())
        rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "Category" not in ln]
        rows = [ln for ln in rows if set(ln.replace("|", "").replace(":", "").strip()) - {"-", ""}]
        assert len(rows) == len(report.results)

    def test_pipes_escaped(self):
        report = {
            "policy_name": "p", "env_id": "E", "overall_passed": True,
            "results": [{"category": "c", "name": "n", "passed": True, "summary": "a | b"}],
        }
        assert "a \\| b" in render_markdown(report)

    def test_empty_results(self):
        md = render_markdown(
            {"policy_name": "p", "env_id": "E", "overall_passed": True, "results": []}
        )
        assert "No test categories were executed" in md
        assert "OVERALL: ✅ **PASS**" in md
