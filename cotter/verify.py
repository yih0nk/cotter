"""Verify a Cotter report's integrity and provenance.

The reproducibility manifest and ``content_sha256`` written into every
report (schema v2+) are only useful if they can be checked. This module
recomputes the content hash (tamper-evidence) and, when a policy file is
supplied, re-hashes it against the manifest's ``policy_sha256``. The CLI
``cotter verify`` wraps it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cotter.manifest import hash_file
from cotter.report import content_hash_of


@dataclass
class Check:
    name: str
    ok: bool | None  # None = not applicable (nothing to check)
    detail: str


@dataclass
class VerifyResult:
    checks: list[Check]

    @property
    def ok(self) -> bool:
        """True unless a check explicitly failed (skipped checks don't fail it)."""
        return all(c.ok is not False for c in self.checks)


def verify_report(payload: dict, policy_path: str | Path | None = None) -> VerifyResult:
    """Check a report dict's content hash and (optionally) its policy hash."""
    checks: list[Check] = []

    stored = payload.get("content_sha256")
    if stored is None:
        checks.append(
            Check(
                "content hash",
                None,
                "report has no content_sha256 (schema v1 predates content hashing)",
            )
        )
    else:
        recomputed = content_hash_of(payload)
        ok = recomputed == stored
        checks.append(
            Check(
                "content hash",
                ok,
                "matches — report is unmodified"
                if ok
                else f"MISMATCH — stored {stored}, recomputed {recomputed}",
            )
        )

    if policy_path is not None:
        manifest = payload.get("manifest") or {}
        expected = manifest.get("policy_sha256")
        if expected is None:
            checks.append(
                Check("policy hash", None, "manifest has no policy_sha256 to check against")
            )
        else:
            actual = hash_file(policy_path)
            if actual is None:
                checks.append(
                    Check("policy hash", False, f"policy file not found: {policy_path}")
                )
            else:
                ok = actual == expected
                checks.append(
                    Check(
                        "policy hash",
                        ok,
                        "matches the manifest — same policy artifact"
                        if ok
                        else f"MISMATCH — manifest {expected}, file {actual}",
                    )
                )

    return VerifyResult(checks)


def load_report(path: str | Path) -> dict:
    """Load a JSON report from disk (raises on missing/invalid file)."""
    return json.loads(Path(path).read_text())
