"""Regulatory compliance document generation — PAID TIER (license required).

This module is a deliberate stub. The open-source Cotter engine produces
the *evidence* (structured TestReport pass/fail results with statistical
guarantees); turning that evidence into signed regulatory documentation
— EU Machinery Regulation 2027 technical files, ISO 10218 conformity
records — is the licensed commercial layer.

The public import path is stable so downstream code and docs can be
written against it today::

    from cotter.compliance import EUMachineryReg2027

    generator = EUMachineryReg2027(report)   # raises LicenseRequiredError

Every constructor here raises :class:`LicenseRequiredError` with
activation instructions. Nothing in this package performs real document
generation in the open-source distribution.
"""

from cotter.compliance.base import (
    ComplianceGenerator,
    LicenseRequiredError,
    EUMachineryReg2027,
    ISO10218,
)

__all__ = [
    "ComplianceGenerator",
    "LicenseRequiredError",
    "EUMachineryReg2027",
    "ISO10218",
]
