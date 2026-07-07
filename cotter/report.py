"""Lightweight structured test report.

:class:`TestReport` aggregates per-category results into a console
summary and a JSON artifact. It is a results container only — compliance
document generation is explicitly out of scope.
"""

from __future__ import annotations

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


@dataclass
class TestReport:
    policy_name: str
    env_id: str
    metadata: dict = field(default_factory=dict)
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
            f"({result.success_rate:.1%}) after {result.n_trials} sequential trials "
            f"(H0 p<={result.p0}, H1 p>={result.p1})",
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
            f"trials (p={result.p_value:.4f}, {result.test})",
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

    def to_dict(self) -> dict:
        return {
            "cotter_report_version": 1,
            "policy_name": self.policy_name,
            "env_id": self.env_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "overall_passed": self.overall_passed,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, cls=_NumpyEncoder) + "\n")
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
