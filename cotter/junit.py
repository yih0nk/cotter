"""Render a Cotter report as JUnit XML.

JUnit XML is the lingua franca of CI test reporting: GitHub Actions,
GitLab, Jenkins, CircleCI and others parse it natively to show per-test
pass/fail without any custom plumbing. Mapping Cotter's categories onto
it makes a Cotter run show up as ordinary test results in those UIs.

Mapping: each category result becomes a ``<testcase>`` under
``classname="cotter.<category>"``. A failed category carries a
``<failure>``; an informational one (``passed is None``) a ``<skipped>``;
a passing one has no child. Built with ``xml.etree`` so every value is
escaped.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

_PROPERTY_KEYS = ("policy_name", "env_id", "content_sha256")


def to_junit_xml(report: dict) -> str:
    """Render a report dict (see ``TestReport.to_dict``) as a JUnit XML string."""
    results = report.get("results", [])
    n = len(results)
    failures = sum(1 for r in results if r.get("passed") is False)
    skipped = sum(1 for r in results if r.get("passed") is None)

    counts = {
        "name": "cotter",
        "tests": str(n),
        "failures": str(failures),
        "errors": "0",
        "skipped": str(skipped),
    }
    suites = ET.Element("testsuites", counts)
    suite_attrs = dict(counts)
    suite_attrs["name"] = f"cotter/{report.get('policy_name', 'policy')}"
    if report.get("created_at"):
        suite_attrs["timestamp"] = report["created_at"]
    suite = ET.SubElement(suites, "testsuite", suite_attrs)

    present = [(k, report.get(k)) for k in _PROPERTY_KEYS if report.get(k)]
    if present:
        props = ET.SubElement(suite, "properties")
        for key, value in present:
            ET.SubElement(props, "property", {"name": key, "value": str(value)})

    for r in results:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": f"cotter.{r.get('category', 'unknown')}",
                "name": str(r.get("name", "unknown")),
                "time": "0",
            },
        )
        summary = str(r.get("summary", ""))
        if r.get("passed") is False:
            failure = ET.SubElement(case, "failure", {"message": summary})
            failure.text = summary
        elif r.get("passed") is None:
            ET.SubElement(
                case, "skipped", {"message": "informational result (no pass/fail semantics)"}
            )

    ET.indent(suites)
    body = ET.tostring(suites, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + body + "\n"
