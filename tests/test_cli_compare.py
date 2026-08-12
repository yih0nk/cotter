"""Tests for the `cotter compare` subcommand."""

from pathlib import Path

import pytest
import torch

from cotter.cli import main

VICTIM = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"

pytestmark = pytest.mark.skipif(not VICTIM.exists(), reason="victim artifact missing")

# A regression config: baseline is a placeholder overridden by --baseline.
COMPARE_CFG = (
    "env: InvertedPendulum-v5\n"
    "success: {type: min_length, value: 100}\n"
    "regression: {baseline: placeholder.zip, n_pairs: 6}\n"
)


class ZeroPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "compare.yaml"
    p.write_text(COMPARE_CFG)
    return p


@pytest.fixture
def weak_candidate(tmp_path):
    path = tmp_path / "weak.pt"
    torch.save(ZeroPolicy(), path)
    return path


class TestCompare:
    def test_self_comparison_no_regression_exits_zero(self, cfg_file, capsys):
        rc = main([
            "compare", "--baseline", str(VICTIM), "--candidate", str(VICTIM),
            "--config", str(cfg_file), "--quiet",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "NO_REGRESSION" in out
        # only regression rows, nothing else
        assert "performance/" not in out
        assert "adversarial/" not in out

    def test_weak_candidate_regression_exits_one(self, cfg_file, weak_candidate, capsys):
        rc = main([
            "compare", "--baseline", str(VICTIM), "--candidate", str(weak_candidate),
            "--config", str(cfg_file), "--quiet",
        ])
        out = capsys.readouterr().out
        assert rc == 1
        assert "REGRESSION" in out

    def test_report_flags_write_json_and_html(self, cfg_file, tmp_path):
        json_path = tmp_path / "out" / "compare.json"
        html_path = tmp_path / "out" / "compare.html"
        rc = main([
            "compare", "--baseline", str(VICTIM), "--candidate", str(VICTIM),
            "--config", str(cfg_file), "--report", str(json_path),
            "--report-html", str(html_path), "--quiet",
        ])
        assert rc == 0
        assert json_path.exists()
        assert '"cotter_report_version"' in json_path.read_text()
        assert html_path.read_text().startswith("<!doctype html>")

    def test_missing_candidate_exits_two(self, cfg_file, tmp_path, capsys):
        rc = main([
            "compare", "--baseline", str(VICTIM),
            "--candidate", str(tmp_path / "ghost.zip"),
            "--config", str(cfg_file), "--quiet",
        ])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_missing_config_exits_two(self, tmp_path, capsys):
        rc = main([
            "compare", "--baseline", str(VICTIM), "--candidate", str(VICTIM),
            "--config", str(tmp_path / "nope.yaml"), "--quiet",
        ])
        assert rc == 2

    def test_requires_baseline_and_candidate(self):
        with pytest.raises(SystemExit) as exc:
            main(["compare", "--config", "x.yaml"])
        assert exc.value.code == 2
