"""Render a Cotter report as GitHub-flavored Markdown.

A shareable plain-text report — paste it into a pull request, an issue, a
wiki, or any CI that renders Markdown. Pure and dependency-free (reads a
report dict), mirroring :mod:`cotter.render` (HTML) and
:mod:`cotter.junit` (JUnit XML).
"""

from __future__ import annotations

_STATUS = {True: "✅", False: "❌", None: "➖"}
_VERDICT = {True: "✅ **PASS**", False: "❌ **FAIL**"}


def _cell(text) -> str:
    """Make an arbitrary value safe inside a Markdown table cell."""
    s = str(text)
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(report: dict) -> str:
    """Render a report dict (see ``TestReport.to_dict``) as a Markdown string."""
    results = report.get("results", [])
    n_pass = sum(1 for r in results if r.get("passed") is True)
    n_fail = sum(1 for r in results if r.get("passed") is False)
    n_info = sum(1 for r in results if r.get("passed") is None)
    overall = bool(report.get("overall_passed"))

    lines = ["# Cotter test report", ""]
    lines.append(
        f"**Policy** `{_cell(report.get('policy_name', '?'))}` on "
        f"`{_cell(report.get('env_id', '?'))}` — OVERALL: {_VERDICT[overall]}"
    )
    lines.append("")
    lines.append(f"`{n_pass}` passing · `{n_fail}` failing · `{n_info}` informational")
    lines.append("")

    if results:
        lines.append("| | Category | Test | Result |")
        lines.append("|:--:|---|---|---|")
        for r in results:
            lines.append(
                f"| {_STATUS.get(r.get('passed'), '➖')} "
                f"| `{_cell(r.get('category', '?'))}` "
                f"| `{_cell(r.get('name', '?'))}` "
                f"| {_cell(r.get('summary', ''))} |"
            )
    else:
        lines.append("_No test categories were executed._")

    manifest = report.get("manifest")
    if manifest:
        lines.append("")
        lines.append("<details><summary>Reproducibility manifest</summary>")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for key, value in manifest.items():
            if isinstance(value, dict):
                value = ", ".join(f"{k} {v}" for k, v in value.items())
            lines.append(f"| `{_cell(key)}` | {_cell(value)} |")
        lines.append("")
        lines.append("</details>")

    footer = []
    if report.get("created_at"):
        footer.append(f"generated {_cell(report['created_at'])}")
    if report.get("content_sha256"):
        footer.append(f"`{_cell(report['content_sha256'])}`")
    if footer:
        lines.append("")
        lines.append("— " + " · ".join(footer))

    lines.append("")
    return "\n".join(lines) + "\n"
