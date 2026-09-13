"""Tests for report-to-report diff (cotter.diff)."""

from cotter.diff import diff_reports


def _report(results):
    return {"results": [{"category": c, "name": n, "passed": p} for c, n, p in results]}


class TestDiffReports:
    def test_regression_detected(self):
        before = _report([("safety", "limits", True)])
        after = _report([("safety", "limits", False)])
        d = diff_reports(before, after)
        assert d.regressed
        assert len(d.regressions) == 1
        assert d.regressions[0].name == "limits"

    def test_improvement_not_a_regression(self):
        d = diff_reports(_report([("perf", "sprt", False)]), _report([("perf", "sprt", True)]))
        assert not d.regressed
        assert len(d.improvements) == 1

    def test_unchanged(self):
        r = _report([("safety", "limits", True)])
        d = diff_reports(r, r)
        assert not d.regressed
        assert d.deltas[0].status == "unchanged"

    def test_added_and_removed(self):
        before = _report([("safety", "a", True)])
        after = _report([("safety", "b", True)])
        d = diff_reports(before, after)
        assert {x.name for x in d.added} == {"b"}
        assert {x.name for x in d.removed} == {"a"}
        assert not d.regressed  # add/remove alone isn't a regression

    def test_informational_to_fail_is_regression(self):
        # None (informational) -> False counts as a regression
        d = diff_reports(_report([("perf", "sprt", None)]), _report([("perf", "sprt", False)]))
        assert d.regressed

    def test_multiple_and_summary(self):
        before = _report([("safety", "a", True), ("perf", "b", True), ("adv", "c", False)])
        after = _report([("safety", "a", False), ("perf", "b", True), ("adv", "c", True)])
        d = diff_reports(before, after)
        assert len(d.regressions) == 1  # a
        assert len(d.improvements) == 1  # c
        assert "REGRESSED" in d.summary()
        assert d.to_dict()["n_regressions"] == 1
