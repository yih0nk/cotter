"""Unit tests for JUnit XML rendering (cotter.junit)."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cotter.junit import to_junit_xml
from cotter.report import TestReport
from cotter.tests.safety import SafetyLimit, evaluate_safety
from cotter.tests.sprt import run_sprt


@pytest.fixture
def report():
    r = TestReport(policy_name="victim", env_id="InvertedPendulum-v5")
    r.add_sprt(run_sprt(iter([True] * 10), p0=0.5, p1=0.9))  # pass
    r.add_safety(evaluate_safety([[{"v": np.array([2.0])}]], [SafetyLimit("v", 1.0)]))  # fail
    return r


def parse(report):
    return ET.fromstring(to_junit_xml(report.to_dict()))


class TestJUnit:
    def test_is_well_formed_xml(self, report):
        xml = to_junit_xml(report.to_dict())
        assert xml.startswith('<?xml version="1.0"')
        ET.fromstring(xml)  # raises if malformed

    def test_suite_counts(self, report):
        root = parse(report)
        assert root.attrib["tests"] == "2"
        assert root.attrib["failures"] == "1"
        assert root.attrib["skipped"] == "0"

    def test_one_testcase_per_result(self, report):
        cases = parse(report).findall(".//testcase")
        assert len(cases) == 2
        classnames = {c.attrib["classname"] for c in cases}
        assert classnames == {"cotter.performance", "cotter.safety"}

    def test_failed_category_has_failure_child(self, report):
        cases = parse(report).findall(".//testcase")
        safety = next(c for c in cases if c.attrib["classname"] == "cotter.safety")
        failure = safety.find("failure")
        assert failure is not None
        assert "FAIL" in failure.attrib["message"]

    def test_passing_category_has_no_children(self, report):
        cases = parse(report).findall(".//testcase")
        perf = next(c for c in cases if c.attrib["classname"] == "cotter.performance")
        assert list(perf) == []

    def test_informational_maps_to_skipped(self):
        r = TestReport(policy_name="p", env_id="E")
        res = run_sprt(iter([True, False] * 5), p0=0.4, p1=0.6, n_max=10)  # inconclusive
        r.add_sprt(res)
        root = ET.fromstring(to_junit_xml(r.to_dict()))
        assert root.attrib["skipped"] == "1"
        assert root.find(".//testcase/skipped") is not None

    def test_properties_include_provenance(self, report):
        report.manifest = {"cotter_version": "0.3.0"}
        root = parse(report)
        names = {p.attrib["name"] for p in root.findall(".//property")}
        assert "policy_name" in names
        assert "content_sha256" in names  # present via to_dict

    def test_special_characters_are_escaped(self):
        r = TestReport(policy_name="a<b>&\"c", env_id="E")
        r.add("safety", "n<ame", False, "bad & <dangerous>", {})
        root = ET.fromstring(to_junit_xml(r.to_dict()))  # must parse
        # values round-trip intact through escaping
        policy_prop = next(
            p for p in root.findall(".//property") if p.attrib["name"] == "policy_name"
        )
        assert policy_prop.attrib["value"] == "a<b>&\"c"
        assert root.find(".//testcase").attrib["name"] == "n<ame"


class TestToJunitMethod:
    def test_writes_file(self, report, tmp_path):
        out = report.to_junit(tmp_path / "sub" / "report.xml")
        assert out.exists()
        ET.parse(out)  # parseable
