"""Tests for the cotter CLI."""

from pathlib import Path

import pytest

from cotter.cli import main

VICTIM = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"


def write_config(tmp_path, body: str) -> Path:
    cfg = tmp_path / "run.yaml"
    cfg.write_text(body)
    return cfg


MINIMAL = (
    "env: InvertedPendulum-v5\n"
    "success: {type: min_length, value: 50}\n"
    "performance: {p0: 0.5, p1: 0.9, n_max: 5}\n"
)


class TestArgumentErrors:
    def test_missing_config_file(self, tmp_path, capsys):
        rc = main(["run", "--policy", str(VICTIM), "--config", str(tmp_path / "nope.yaml")])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_invalid_config(self, tmp_path, capsys):
        cfg = write_config(tmp_path, "env: X\n")  # missing success
        rc = main(["run", "--policy", str(VICTIM), "--config", str(cfg)])
        assert rc == 2
        assert "must set 'success'" in capsys.readouterr().err

    def test_missing_policy_file(self, tmp_path, capsys):
        cfg = write_config(tmp_path, MINIMAL)
        rc = main(["run", "--policy", str(tmp_path / "ghost.zip"), "--config", str(cfg)])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_run_requires_policy_and_config(self):
        with pytest.raises(SystemExit) as exc:
            main(["run"])
        assert exc.value.code == 2


@pytest.mark.skipif(not VICTIM.exists(), reason="victim artifact missing")
class TestRunCommand:
    def test_passing_run_exits_zero(self, tmp_path, capsys):
        cfg = write_config(tmp_path, MINIMAL)
        rc = main(["run", "--policy", str(VICTIM), "--config", str(cfg), "--quiet"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "COTTER TEST REPORT" in out
        assert "OVERALL: PASS" in out

    def test_failing_run_exits_one(self, tmp_path, capsys):
        # An unreachable success bar (min_return 1e9) makes SPRT FAIL fast.
        cfg = write_config(
            tmp_path,
            "env: InvertedPendulum-v5\n"
            "success: {type: min_return, value: 1000000000}\n"
            "performance: {p0: 0.5, p1: 0.9, n_max: 5}\n",
        )
        rc = main(["run", "--policy", str(VICTIM), "--config", str(cfg), "--quiet"])
        assert rc == 1
        assert "OVERALL: FAIL" in capsys.readouterr().out

    def test_env_override_mismatch_fails_loudly(self, tmp_path, capsys):
        # Overriding to an env with different spaces must exit 2 with a
        # space-mismatch message, not crash mid-rollout. The mismatch is
        # caught either by SB3 at load time (env is passed for HER support)
        # or by cotter's own validate_spaces probe; both are loud.
        cfg = write_config(tmp_path, MINIMAL)
        rc = main([
            "run", "--policy", str(VICTIM), "--config", str(cfg),
            "--env", "HalfCheetah-v5", "--quiet",
        ])
        err = capsys.readouterr().err
        assert rc == 2
        assert "spaces do not match" in err or "trained on observations" in err

    def test_report_override(self, tmp_path):
        cfg = write_config(tmp_path, MINIMAL)
        report_path = tmp_path / "custom" / "r.json"
        rc = main([
            "run", "--policy", str(VICTIM), "--config", str(cfg),
            "--report", str(report_path), "--quiet",
        ])
        assert rc == 0
        assert report_path.exists()

    def test_report_html_override(self, tmp_path):
        cfg = write_config(tmp_path, MINIMAL)
        html_path = tmp_path / "out" / "r.html"
        rc = main([
            "run", "--policy", str(VICTIM), "--config", str(cfg),
            "--report-html", str(html_path), "--quiet",
        ])
        assert rc == 0
        assert html_path.exists()
        assert html_path.read_text().startswith("<!doctype html>")

    def test_run_report_carries_manifest_and_content_hash(self, tmp_path):
        import json

        cfg = write_config(tmp_path, MINIMAL)
        report_path = tmp_path / "r.json"
        rc = main([
            "run", "--policy", str(VICTIM), "--config", str(cfg),
            "--report", str(report_path), "--quiet",
        ])
        assert rc == 0
        report = json.loads(report_path.read_text())
        assert report["cotter_report_version"] == 2
        m = report["manifest"]
        assert m["cotter_version"]  # non-empty
        assert m["env_id"] == "InvertedPendulum-v5"
        assert m["policy_sha256"].startswith("sha256:")
        assert report["content_sha256"].startswith("sha256:")
