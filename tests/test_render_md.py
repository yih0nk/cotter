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


class TestManifestAndHash:
    def test_manifest_rendered_when_present(self, report):
        report.manifest = {"cotter_version": "0.3.0", "dependencies": {"torch": "2.4.1"}}
        md = render_markdown(report.to_dict())
        assert "Reproducibility manifest" in md
        assert "0.3.0" in md
        # nested dict flattened, not a raw python dict
        assert "torch 2.4.1" in md
        assert "{'" not in md

    def test_manifest_absent_when_empty(self, report):
        assert not report.manifest
        assert "Reproducibility manifest" not in render_markdown(report.to_dict())

    def test_content_hash_in_footer(self, report):
        md = render_markdown(report.to_dict())
        assert report.content_hash() in md


class TestToMarkdownMethod:
    def test_writes_file_matching_renderer(self, report, tmp_path):
        out = report.to_markdown(tmp_path / "sub" / "report.md")
        assert out.exists()
        assert out.read_text() == render_markdown(report.to_dict())
