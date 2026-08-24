"""Lightweight structured test report.

:class:`TestReport` aggregates per-category results into a console
summary and a JSON artifact. It is a results container only — compliance
document generation is explicitly out of scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cotter.tests.regression import RegressionDecision, RegressionResult
from cotter.tests.safety import SafetyDecision, SafetyResult
from cotter.tests.sprt import SPRTDecision, SPRTResult


@dataclass
class CategoryResult:
    category: str  # performance | safety | regression | adversarial
    name: str
    passed: bool | None  # None = informational, no pass/fail semantics
    summary: str
    data: dict

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "passed": self.passed,
            "summary": self.summary,
            "data": self.data,
        }


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Fields excluded from the content hash: the timestamp varies run-to-run
# and the hash cannot cover itself.
_HASH_EXCLUDED = ("created_at", "content_sha256")


def content_hash_of(payload: dict) -> str:
    """``"sha256:<hex>"`` digest over a report body.

    Works on any report dict — including one loaded from disk — so a
    stored ``content_sha256`` can be independently recomputed and checked
    (see ``cotter verify``). Excludes the timestamp and the hash field
    itself and sorts keys, so the digest is deterministic.
    """
    body = {k: v for k, v in payload.items() if k not in _HASH_EXCLUDED}
    canonical = json.dumps(body, sort_keys=True, cls=_NumpyEncoder)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class TestReport:
    policy_name: str
    env_id: str
    metadata: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    results: list[CategoryResult] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def add(
        self,
        category: str,
        name: str,
        passed: bool | None,
        summary: str,
        data: dict,
    ) -> None:
        self.results.append(
            CategoryResult(category=category, name=name, passed=passed, summary=summary, data=data)
        )

    # --- adapters for cotter's built-in test result types ---------------

    def add_sprt(self, result: SPRTResult, name: str = "sprt_success_rate") -> None:
        passed = {
            SPRTDecision.PASS: True,
            SPRTDecision.FAIL: False,
            SPRTDecision.INCONCLUSIVE: None,
        }[result.decision]
        self.add(
            "performance",
            name,
            passed,
            f"{result.decision.value}: {result.n_successes}/{result.n_trials} successes "
            f"({result.success_rate:.1%}, {result.ci_level:.0%} CI "
            f"[{result.ci_lower:.1%}, {result.ci_upper:.1%}]) after {result.n_trials} "
            f"sequential trials (H0 p<={result.p0}, H1 p>={result.p1})",
            result.to_dict(),
        )

    def add_safety(self, result: SafetyResult, name: str = "hard_limits") -> None:
        if result.decision == SafetyDecision.PASS:
            summary = (
                f"PASS: no violations in {result.n_timesteps_checked} timesteps "
                f"across {result.n_trials} trials"
            )
        else:
            v = result.violations[0]
            summary = (
                f"FAIL: {result.n_violations} violation(s); first at trial {v.trial} "
                f"step {v.timestep}: |{v.quantity}|={abs(v.value):.3f} > {v.limit}"
            )
        self.add("safety", name, result.decision == SafetyDecision.PASS, summary, result.to_dict())

    def add_regression(self, result: RegressionResult, name: str = "vs_baseline") -> None:
        self.add(
            "regression",
            name,
            result.decision != RegressionDecision.REGRESSION,
            f"{result.decision.value}: baseline {result.baseline_metric:.3f} vs "
            f"candidate {result.candidate_metric:.3f} over {result.n_pairs} paired "
            f"trials (p={result.p_value:.3g}, {result.effect_size_name}="
            f"{result.effect_size:.3g}, {result.test})",
            result.to_dict(),
        )

    def add_adversarial(self, result, name: str = "observation_perturbation") -> None:
        """Adapter for cotter.tests.adversarial.AdversarialResult."""
        self.add("adversarial", name, result.passed, result.summary(), result.to_dict())

    # --- output ----------------------------------------------------------

    @property
    def overall_passed(self) -> bool:
        """True iff no category explicitly failed (informational results ignored)."""
        return all(r.passed is not False for r in self.results)

    def _content_payload(self) -> dict:
        """The report body that the content hash covers.

        Deliberately excludes the timestamp and the hash itself, so the
        hash is deterministic and independently verifiable: recompute
        sha256 over ``to_dict()`` minus ``created_at`` and
        ``content_sha256`` and it must match.
        """
        return {
            "cotter_report_version": 2,
            "policy_name": self.policy_name,
            "env_id": self.env_id,
            "metadata": self.metadata,
            "manifest": self.manifest,
            "overall_passed": self.overall_passed,
            "results": [r.to_dict() for r in self.results],
        }

    def content_hash(self) -> str:
        """A ``"sha256:<hex>"`` digest over the report body (tamper-evident)."""
        return content_hash_of(self._content_payload())

    def to_dict(self) -> dict:
        payload = self._content_payload()
        payload["created_at"] = self.created_at
        payload["content_sha256"] = self.content_hash()
        return payload

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, cls=_NumpyEncoder) + "\n")
        return path

    def to_html(self, path: str | Path) -> Path:
        """Write a self-contained, shareable HTML report (free tier).

        The rendering lives in :mod:`cotter.render`; imported lazily so
        the report container stays dependency-light for JSON-only use.
        """
        from cotter.render import render_html

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(self))
        return path

    def to_junit(self, path: str | Path) -> Path:
        """Write a JUnit XML report for native rendering in CI systems."""
        from cotter.junit import to_junit_xml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_junit_xml(self.to_dict()))
        return path

    def summary(self) -> str:
        width = 78
        mark = {True: "PASS", False: "FAIL", None: "INFO"}
        lines = [
            "=" * width,
            f"COTTER TEST REPORT — policy '{self.policy_name}' on {self.env_id}",
            f"generated {self.created_at}",
            "=" * width,
        ]
        for r in self.results:
            lines.append(f"[{mark[r.passed]:>4}] {r.category}/{r.name}")
            lines.append(f"       {r.summary}")
        lines.append("-" * width)
        lines.append(
            f"OVERALL: {'PASS' if self.overall_passed else 'FAIL'} "
            f"({sum(1 for r in self.results if r.passed is False)} failing, "
            f"{sum(1 for r in self.results if r.passed is True)} passing, "
            f"{sum(1 for r in self.results if r.passed is None)} informational)"
        )
        lines.append("=" * width)
        return "\n".join(lines)
