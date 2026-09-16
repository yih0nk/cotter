"""Requirement/clause traceability.

Conformity assessment is clause-by-clause: an auditor asks "which
requirement does this test evidence, and which requirements are *not*
verified?" This maps each declared clause (an EU Machinery Regulation
EHSR, an ISO 10218 / TS 15066 clause, an EU AI Act article, …) to the
checks that evidence it, and reports each clause as verified, failed, or
unverified — the coverage-gap view an auditor wants.

The clause→limit *values* (the certified thresholds) are the paid layer;
this open mapping just ties existing check results to the clauses the
user declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ClauseCoverage:
    clause: str
    description: str
    status: str  # verified | failed | unverified
    evidence: list[str]  # matching "category/name" check keys

    def to_dict(self) -> dict:
        return {
            "clause": self.clause,
            "description": self.description,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass
class TraceabilityResult:
    clauses: list[ClauseCoverage] = field(default_factory=list)

    def _of(self, status: str) -> list[ClauseCoverage]:
        return [c for c in self.clauses if c.status == status]

    @property
    def verified(self) -> list[ClauseCoverage]:
        return self._of("verified")

    @property
    def failed(self) -> list[ClauseCoverage]:
        return self._of("failed")

    @property
    def unverified(self) -> list[ClauseCoverage]:
        return self._of("unverified")

    @property
    def all_verified(self) -> bool:
        """True iff every clause has passing evidence and none failed."""
        return not self.failed and not self.unverified

    def to_dict(self) -> dict:
        return {
            "all_verified": self.all_verified,
            "n_verified": len(self.verified),
            "n_failed": len(self.failed),
            "n_unverified": len(self.unverified),
            "clauses": [c.to_dict() for c in self.clauses],
        }

    def summary(self) -> str:
        verdict = "COMPLETE" if self.all_verified else "GAPS"
        return (
            f"{verdict}: {len(self.verified)} verified, {len(self.failed)} failed, "
            f"{len(self.unverified)} unverified across {len(self.clauses)} clauses"
        )


def _matches(result, patterns: list[str]) -> bool:
    """A check matches a clause if its name or 'category/name' is listed."""
    name = getattr(result, "name", "")
    category = getattr(result, "category", "")
    keys = {name, f"{category}/{name}"}
    return any(p in keys for p in patterns)


def build_traceability(results: Iterable, mapping: Mapping[str, dict]) -> TraceabilityResult:
    """Map declared clauses to the check results that evidence them.

    ``results`` are report entries (each with ``category``, ``name``,
    ``passed``). ``mapping`` maps a clause id to
    ``{"description": str, "checks": [check name or "category/name", …]}``.
    A clause is *failed* if any matching check failed, *verified* if it has
    a passing check and none failed, else *unverified* (no evidence, or
    only informational results).
    """
    results = list(results)
    clauses: list[ClauseCoverage] = []
    for clause, spec in mapping.items():
        patterns = list(spec.get("checks", []))
        matched = [r for r in results if _matches(r, patterns)]
        keys = [f"{r.category}/{r.name}" for r in matched]
        failing = [r for r in matched if r.passed is False]
        passing = [r for r in matched if r.passed is True]
        if failing:
            status = "failed"
        elif passing:
            status = "verified"
        else:
            status = "unverified"
        clauses.append(
            ClauseCoverage(clause, spec.get("description", ""), status, keys)
        )
    return TraceabilityResult(clauses=clauses)
