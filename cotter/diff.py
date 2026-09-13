"""Compare two Cotter reports — a safety-budget regression gate.

"Did this policy version break any check that used to pass?" is the most
useful question in a retrain loop. :func:`diff_reports` matches checks
across two reports by ``(category, name)`` and classifies each as
regressed (was passing, now failing), improved, unchanged, added, or
removed. A regression is the gate: ``cotter diff`` exits non-zero when
any previously-passing check now fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckDelta:
    category: str
    name: str
    status: str  # regressed | improved | unchanged | added | removed
    before: bool | None
    after: bool | None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "status": self.status,
            "before": self.before,
            "after": self.after,
        }


def _passed_by_key(report: dict) -> dict[tuple[str, str], bool | None]:
    out: dict[tuple[str, str], bool | None] = {}
    for r in report.get("results", []):
        out[(r.get("category", "?"), r.get("name", "?"))] = r.get("passed")
    return out


@dataclass
class DiffResult:
    deltas: list[CheckDelta] = field(default_factory=list)

    def _of(self, status: str) -> list[CheckDelta]:
        return [d for d in self.deltas if d.status == status]

    @property
    def regressions(self) -> list[CheckDelta]:
        return self._of("regressed")

    @property
    def improvements(self) -> list[CheckDelta]:
        return self._of("improved")

    @property
    def added(self) -> list[CheckDelta]:
        return self._of("added")

    @property
    def removed(self) -> list[CheckDelta]:
        return self._of("removed")

    @property
    def regressed(self) -> bool:
        """True iff any previously-passing check now fails (the gate)."""
        return bool(self.regressions)

    def to_dict(self) -> dict:
        return {
            "regressed": self.regressed,
            "n_regressions": len(self.regressions),
            "n_improvements": len(self.improvements),
            "n_added": len(self.added),
            "n_removed": len(self.removed),
            "deltas": [d.to_dict() for d in self.deltas],
        }

    def summary(self) -> str:
        verdict = "REGRESSED" if self.regressed else "no regression"
        return (
            f"{verdict}: {len(self.regressions)} regressed, "
            f"{len(self.improvements)} improved, {len(self.added)} added, "
            f"{len(self.removed)} removed"
        )


def diff_reports(before: dict, after: dict) -> DiffResult:
    """Diff two report dicts, classifying every check by ``(category, name)``."""
    before_map = _passed_by_key(before)
    after_map = _passed_by_key(after)

    deltas: list[CheckDelta] = []
    for key in sorted(before_map.keys() | after_map.keys()):
        category, name = key
        in_before, in_after = key in before_map, key in after_map
        b = before_map.get(key)
        a = after_map.get(key)
        if in_before and not in_after:
            status = "removed"
        elif in_after and not in_before:
            status = "added"
        elif b is not False and a is False:
            status = "regressed"  # was passing or informational, now failing
        elif b is False and a is True:
            status = "improved"
        else:
            status = "unchanged"
        deltas.append(CheckDelta(category, name, status, b, a))

    return DiffResult(deltas=deltas)


def load_report(path: str | Path) -> dict:
    """Load a JSON report from disk (raises on missing/invalid file)."""
    return json.loads(Path(path).read_text())
