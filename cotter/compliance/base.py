"""Paid-tier compliance generators — stub implementations.

# paid tier — license required
#
# These classes define the commercial API surface but perform no real
# work in the open-source distribution: constructing any generator
# raises LicenseRequiredError. The shapes here (standard name, version,
# required category coverage) are intentionally concrete so the boundary
# between the free engine and the paid layer is legible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cotter.report import TestReport

_ACTIVATION_HINT = (
    "Regulatory document generation is a licensed feature and is not "
    "included in the open-source Cotter distribution. The open engine "
    "produces the pass/fail evidence (TestReport); the compliance layer "
    "renders it into a regulator-ready technical file. Request access at "
    "https://github.com/yih0nk/cotter (see the commercial tier) or set a "
    "valid COTTER_LICENSE_KEY."
)


class LicenseRequiredError(RuntimeError):
    """Raised when a paid-tier compliance feature is used without a license."""

    def __init__(self, standard: str) -> None:
        super().__init__(
            f"'{standard}' compliance generation requires a Cotter license. "
            f"{_ACTIVATION_HINT}"
        )
        self.standard = standard


class ComplianceGenerator:
    """Base class for regulatory document generators (paid tier).

    Subclasses declare the standard they cover and the test categories
    that standard's evidence requires. The open-source stub validates
    nothing and generates nothing — it refuses at construction.
    """

    standard: str = "unknown"
    version: str = "0"
    required_categories: tuple[str, ...] = ()

    def __init__(self, report: "TestReport") -> None:
        # The license check happens before any work, so downstream code
        # gets a clear, immediate signal rather than a late failure.
        raise LicenseRequiredError(self.standard)

    def generate(self, output_path) -> None:  # pragma: no cover - unreachable in OSS
        raise LicenseRequiredError(self.standard)


class EUMachineryReg2027(ComplianceGenerator):
    """EU Machinery Regulation (2023/1230) technical file generator (paid)."""

    standard = "EU Machinery Regulation 2027"
    version = "2023/1230"
    required_categories = ("performance", "safety", "adversarial")


class ISO10218(ComplianceGenerator):
    """ISO 10218 industrial-robot safety conformity generator (paid)."""

    standard = "ISO 10218"
    version = "2011"
    required_categories = ("safety", "performance")
