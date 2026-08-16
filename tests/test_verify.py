"""Tests for report verification (cotter.verify) and `cotter verify`."""

import json

import pytest

from cotter.cli import main
from cotter.manifest import build_manifest
from cotter.report import TestReport
from cotter.tests.sprt import run_sprt
from cotter.verify import verify_report


def make_report(tmp_path, *, with_policy=None):
    r = TestReport(
        policy_name="p",
        env_id="InvertedPendulum-v5",
        manifest=build_manifest(cotter_version="0.3.0", policy_path=with_policy),
    )
    r.add_sprt(run_sprt(iter([True] * 10), p0=0.5, p1=0.9))
    path = tmp_path / "report.json"
    r.to_json(path)
    return path


class TestVerifyReport:
    def test_clean_report_verifies(self, tmp_path):
        payload = json.loads(make_report(tmp_path).read_text())
        result = verify_report(payload)
        assert result.ok
        assert any(c.name == "content hash" and c.ok for c in result.checks)

    def test_tampered_report_fails(self, tmp_path):
        payload = json.loads(make_report(tmp_path).read_text())
        payload["results"][0]["passed"] = False  # flip a verdict
        result = verify_report(payload)
        assert not result.ok

    def test_v1_report_has_nothing_to_check(self, tmp_path):
        # a report without content_sha256 (older schema)
        result = verify_report({"policy_name": "p", "results": []})
        assert result.ok  # skipped checks don't fail it
        assert result.checks[0].ok is None

    def test_policy_hash_match(self, tmp_path):
        policy = tmp_path / "victim.zip"
        policy.write_bytes(b"weights")
        payload = json.loads(make_report(tmp_path, with_policy=policy).read_text())
        result = verify_report(payload, policy_path=policy)
        assert result.ok
        assert any(c.name == "policy hash" and c.ok for c in result.checks)

    def test_policy_hash_mismatch(self, tmp_path):
        policy = tmp_path / "victim.zip"
        policy.write_bytes(b"weights")
        payload = json.loads(make_report(tmp_path, with_policy=policy).read_text())
        policy.write_bytes(b"tampered weights")  # change the file after the report
        result = verify_report(payload, policy_path=policy)
        assert not result.ok


class TestVerifyCli:
    def test_verify_clean_exits_zero(self, tmp_path, capsys):
        report = make_report(tmp_path)
        rc = main(["verify", str(report)])
        assert rc == 0
        assert "VERIFIED" in capsys.readouterr().out

    def test_verify_tampered_exits_one(self, tmp_path, capsys):
        report = make_report(tmp_path)
        payload = json.loads(report.read_text())
        payload["overall_passed"] = not payload["overall_passed"]
        report.write_text(json.dumps(payload))
        rc = main(["verify", str(report)])
        assert rc == 1
        assert "FAILED" in capsys.readouterr().out

    def test_missing_report_exits_two(self, tmp_path, capsys):
        rc = main(["verify", str(tmp_path / "nope.json")])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_invalid_json_exits_two(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        rc = main(["verify", str(bad)])
        assert rc == 2

    def test_verify_with_policy_flag(self, tmp_path, capsys):
        policy = tmp_path / "victim.zip"
        policy.write_bytes(b"weights")
        report = make_report(tmp_path, with_policy=policy)
        rc = main(["verify", str(report), "--policy", str(policy)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "policy hash" in out
