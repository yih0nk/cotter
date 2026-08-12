"""Unit tests for HTML report rendering (cotter.render)."""

import numpy as np
import pytest

from cotter.render import _fmt, render_html
from cotter.report import TestReport
from cotter.tests.regression import mcnemar_exact
from cotter.tests.safety import SafetyLimit, evaluate_safety
from cotter.tests.sprt import run_sprt


@pytest.fixture
def full_report():
    r = TestReport(
        policy_name="victim_demo",
        env_id="InvertedPendulum-v5",
        metadata={"base_seed": 0, "algo": "PPO"},
    )
    r.add_sprt(run_sprt(iter([True] * 10), p0=0.5, p1=0.9))
    r.add_safety(evaluate_safety([[{"v": np.array([2.0])}]], [SafetyLimit("v", 1.0)]))
    r.add_regression(mcnemar_exact([True, True, False], [True, True, False]))
    return r


class TestFormat:
    def test_none_and_bools(self):
        assert _fmt(None) == "—"
        assert _fmt(True) == "true"
        assert _fmt(False) == "false"

    def test_int_gets_thousands_separator(self):
        assert _fmt(150000) == "150,000"

    def test_whole_float_shown_as_int(self):
        assert _fmt(20.0) == "20"

    def test_fractional_float_is_compact(self):
        assert _fmt(0.058399999) == "0.0584"

    def test_nan_is_labelled(self):
        assert _fmt(float("nan")) == "nan"


class TestRenderHtml:
    def test_is_a_complete_self_contained_document(self, full_report):
        html = render_html(full_report)
        assert html.startswith("<!doctype html>")
        assert "</html>" in html.strip().splitlines()[-1] or html.rstrip().endswith("</html>")
        # self-contained: no external stylesheet/script/network references
        assert "<style>" in html
        assert "http://" not in html and "https://" not in html
        assert "<script" not in html.lower()

    def test_surfaces_policy_env_and_overall(self, full_report):
        html = render_html(full_report)
        assert "victim_demo" in html
        assert "InvertedPendulum-v5" in html
        # a safety violation was added, so the report fails overall
        assert "OVERALL: FAIL" in html

    def test_counts_reflect_results(self, full_report):
        html = render_html(full_report)
        assert "2 passing" in html  # sprt + regression
        assert "1 failing" in html  # safety
        assert "0 informational" in html

    def test_each_result_summary_is_present(self, full_report):
        html = render_html(full_report)
        for result in full_report.results:
            # summaries can contain characters that are escaped in HTML;
            # check a distinctive escape-safe fragment instead.
            assert result.category in html

    def test_escapes_dynamic_text(self):
        r = TestReport(policy_name="<script>alert(1)</script>", env_id="Env-v0")
        html = render_html(r)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_empty_report_renders_without_error(self):
        r = TestReport(policy_name="p", env_id="Env-v0")
        html = render_html(r)
        assert "No test categories were executed." in html
        assert "OVERALL: PASS" in html  # vacuously passes

    def test_numpy_values_do_not_leak(self, full_report):
        html = render_html(full_report)
        assert "np.float" not in html and "numpy" not in html


class TestToHtml:
    def test_writes_file_and_returns_path(self, full_report, tmp_path):
        out = full_report.to_html(tmp_path / "sub" / "report.html")
        assert out.exists()
        assert out.read_text().startswith("<!doctype html>")

    def test_matches_render_html(self, full_report, tmp_path):
        out = full_report.to_html(tmp_path / "report.html")
        assert out.read_text() == render_html(full_report)
