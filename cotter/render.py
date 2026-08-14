"""Standalone HTML rendering for a :class:`~cotter.report.TestReport`.

The paid tier turns a report into a signed, regulator-ready technical
file; this module is the *free* counterpart — a single self-contained
HTML page an engineer can open in a browser, drop into a pull request,
or attach to a demo. No external assets, no JavaScript, no network: the
output is one file that renders the same offline and adapts to the
viewer's light/dark theme.

Everything dynamic is HTML-escaped, so arbitrary policy names, env ids,
and metadata cannot break (or inject into) the page.
"""

from __future__ import annotations

import json
from html import escape

from cotter.report import TestReport, _NumpyEncoder

_BADGE = {True: ("PASS", "pass"), False: ("FAIL", "fail"), None: ("INFO", "info")}


def _fmt(value) -> str:
    """Format a scalar for display (thousands separators, tidy floats)."""
    if value is None:
        return "—"  # em dash
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "∞" if value > 0 else "-∞"
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        return f"{value:.4g}"
    return str(value)


def _render_value(value) -> str:
    """Render a JSON-ish value as escaped HTML (recurses into dicts/lists)."""
    if isinstance(value, dict):
        return _kv_table(value) if value else "<span class='muted'>—</span>"
    if isinstance(value, list):
        if not value:
            return "<span class='muted'>none</span>"
        if all(isinstance(item, dict) for item in value):
            return _rows_table(value)
        return escape(", ".join(_fmt(item) for item in value))
    return f"<span class='val'>{escape(_fmt(value))}</span>"


def _kv_table(data: dict) -> str:
    """A two-column key/value table."""
    rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{_render_value(val)}</td></tr>"
        for key, val in data.items()
    )
    return f"<table class='kv'>{rows}</table>"


def _rows_table(rows: list[dict]) -> str:
    """A grid table for a list of uniform-ish dicts (limits, violations)."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{escape(str(col))}</th>" for col in columns)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{_render_value(row.get(col))}</td>" for col in columns)
        body += f"<tr>{cells}</tr>"
    return f"<table class='grid'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _result_card(result: dict) -> str:
    label, cls = _BADGE[result["passed"]]
    title = f"{escape(result['category'])} / {escape(result['name'])}"
    summary = escape(result["summary"])
    details = _kv_table(result["data"]) if result.get("data") else ""
    details_block = (
        f"<details><summary>Details</summary>{details}</details>" if details else ""
    )
    return (
        f"<section class='card {cls}'>"
        f"<div class='card-head'>"
        f"<span class='badge {cls}'>{label}</span>"
        f"<span class='card-title'>{title}</span>"
        f"</div>"
        f"<p class='summary'>{summary}</p>"
        f"{details_block}"
        f"</section>"
    )


def render_html(report: TestReport) -> str:
    """Render a :class:`TestReport` as a complete, self-contained HTML page."""
    # Round-trip through the numpy-aware encoder so the renderer only ever
    # sees native Python types — no numpy scalars leak into formatting.
    payload = json.loads(json.dumps(report.to_dict(), cls=_NumpyEncoder))

    results = payload["results"]
    n_pass = sum(1 for r in results if r["passed"] is True)
    n_fail = sum(1 for r in results if r["passed"] is False)
    n_info = sum(1 for r in results if r["passed"] is None)
    overall_label, overall_cls = ("PASS", "pass") if payload["overall_passed"] else ("FAIL", "fail")

    meta_block = ""
    if payload["metadata"]:
        meta_block = (
            f"<section class='meta'><h2>Run metadata</h2>"
            f"{_kv_table(payload['metadata'])}</section>"
        )

    manifest_block = ""
    if payload.get("manifest"):
        manifest_block = (
            f"<section class='meta'><h2>Reproducibility manifest</h2>"
            f"{_kv_table(payload['manifest'])}</section>"
        )

    cards = "".join(_result_card(r) for r in results) or (
        "<p class='muted'>No test categories were executed.</p>"
    )

    content_hash = payload.get("content_sha256")
    integrity = (
        f" &middot; <span class='val'>{escape(content_hash)}</span>" if content_hash else ""
    )

    return _PAGE.format(
        title=escape(f"Cotter report — {report.policy_name}"),
        policy=escape(payload["policy_name"]),
        env=escape(payload["env_id"]),
        created=escape(payload["created_at"]),
        version=payload["cotter_report_version"],
        overall_label=overall_label,
        overall_cls=overall_cls,
        n_pass=n_pass,
        n_fail=n_fail,
        n_info=n_info,
        meta_block=meta_block,
        manifest_block=manifest_block,
        integrity=integrity,
        cards=cards,
        css=_CSS,
    )


_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1d21; --muted: #6b7280; --card: #f7f8fa;
  --border: #e3e6ea; --accent: #2563eb;
  --pass: #16a34a; --pass-bg: #e9f7ee;
  --fail: #dc2626; --fail-bg: #fdeaea;
  --info: #6b7280; --info-bg: #eef0f3;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171a; --fg: #e6e8ea; --muted: #9aa3ad; --card: #1c2024;
    --border: #2c3238; --accent: #60a5fa;
    --pass: #4ade80; --pass-bg: #16261c;
    --fail: #f87171; --fail-bg: #2a1717;
    --info: #9aa3ad; --info-bg: #23282d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
header.report { border-bottom: 1px solid var(--border); padding-bottom: 1.25rem; margin-bottom: 1.5rem; }
header.report h1 { margin: 0 0 .35rem; font-size: 1.4rem; letter-spacing: -.01em; }
header.report .sub { color: var(--muted); font-size: .92rem; }
header.report .sub code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--fg); }
.overall { display: inline-flex; align-items: center; gap: .6rem; margin-top: 1rem;
  padding: .5rem .9rem; border-radius: 8px; font-weight: 600; }
.overall.pass { background: var(--pass-bg); color: var(--pass); }
.overall.fail { background: var(--fail-bg); color: var(--fail); }
.overall .counts { font-weight: 400; color: var(--muted); font-size: .85rem; }
h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 2rem 0 .6rem; }
.card { background: var(--card); border: 1px solid var(--border); border-left-width: 3px;
  border-radius: 8px; padding: 1rem 1.1rem; margin-bottom: .9rem; }
.card.pass { border-left-color: var(--pass); }
.card.fail { border-left-color: var(--fail); }
.card.info { border-left-color: var(--info); }
.card-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; }
.card-title { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9rem; color: var(--muted); }
.badge { font-size: .72rem; font-weight: 700; letter-spacing: .04em; padding: .15rem .45rem; border-radius: 5px; }
.badge.pass { background: var(--pass-bg); color: var(--pass); }
.badge.fail { background: var(--fail-bg); color: var(--fail); }
.badge.info { background: var(--info-bg); color: var(--info); }
.summary { margin: .25rem 0 .6rem; }
details { margin-top: .5rem; }
summary { cursor: pointer; color: var(--accent); font-size: .85rem; user-select: none; }
summary:hover { text-decoration: underline; }
.tablewrap, details { overflow-x: auto; }
table { border-collapse: collapse; margin-top: .55rem; font-size: .85rem; width: 100%; }
table.kv th { text-align: left; font-weight: 500; color: var(--muted); vertical-align: top;
  padding: .28rem .8rem .28rem 0; white-space: nowrap; }
table.kv td { padding: .28rem 0; }
table.grid th, table.grid td { border: 1px solid var(--border); padding: .3rem .55rem; text-align: left; }
table.grid th { color: var(--muted); font-weight: 500; }
.val { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.muted { color: var(--muted); }
footer { margin-top: 2.5rem; color: var(--muted); font-size: .8rem; text-align: center; }
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
<header class="report">
<h1>Cotter test report</h1>
<div class="sub">policy <code>{policy}</code> on <code>{env}</code> &middot; generated {created}</div>
<div class="overall {overall_cls}">
<span>OVERALL: {overall_label}</span>
<span class="counts">{n_pass} passing &middot; {n_fail} failing &middot; {n_info} informational</span>
</div>
</header>
{meta_block}
<h2>Results</h2>
{cards}
{manifest_block}
<footer>Generated by Cotter &middot; report schema v{version}{integrity}. The open engine emits the evidence; the compliance layer renders the certified technical file.</footer>
</div>
</body>
</html>
"""
