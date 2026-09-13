"""Tests for the `cotter diff` CLI."""

import json

from cotter.cli import main


def _write(path, results):
    path.write_text(json.dumps(
        {"results": [{"category": c, "name": n, "passed": p} for c, n, p in results]}
    ))
    return path


class TestCliDiff:
    def test_no_regression_exits_zero(self, tmp_path, capsys):
        a = _write(tmp_path / "a.json", [("safety", "limits", True)])
        b = _write(tmp_path / "b.json", [("safety", "limits", True)])
        rc = main(["diff", str(a), str(b)])
        assert rc == 0
        assert "no regression" in capsys.readouterr().out

    def test_regression_exits_one(self, tmp_path, capsys):
        a = _write(tmp_path / "a.json", [("safety", "limits", True)])
        b = _write(tmp_path / "b.json", [("safety", "limits", False)])
        rc = main(["diff", str(a), str(b)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "REGRESSED" in out
        assert "safety/limits" in out

    def test_missing_report_exits_two(self, tmp_path, capsys):
        a = _write(tmp_path / "a.json", [("safety", "limits", True)])
        rc = main(["diff", str(a), str(tmp_path / "nope.json")])
        assert rc == 2
        assert "not found" in capsys.readouterr().err

    def test_invalid_json_exits_two(self, tmp_path):
        a = _write(tmp_path / "a.json", [("safety", "limits", True)])
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert main(["diff", str(a), str(bad)]) == 2
