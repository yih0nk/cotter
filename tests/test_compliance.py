"""Tests for the paid-tier compliance stub.

The contract under test: the import paths are stable and every generator
refuses at construction with an actionable license error. Nothing here
generates a real document.
"""

import pytest

from cotter.compliance import (
    ComplianceGenerator,
    EUMachineryReg2027,
    ISO10218,
    LicenseRequiredError,
)
from cotter.report import TestReport


@pytest.fixture
def report():
    return TestReport(policy_name="p", env_id="InvertedPendulum-v5")


class TestImportSurface:
    def test_expected_names_importable(self):
        # The paid-tier API path must be stable for downstream code today.
        from cotter.compliance import EUMachineryReg2027 as _euref  # noqa: F401

    def test_generators_declare_standard_metadata(self):
        assert EUMachineryReg2027.standard == "EU Machinery Regulation 2027"
        assert "safety" in EUMachineryReg2027.required_categories
        assert ISO10218.standard == "ISO 10218"


class TestLicenseGating:
    @pytest.mark.parametrize("cls", [EUMachineryReg2027, ISO10218])
    def test_construction_raises_license_error(self, cls, report):
        with pytest.raises(LicenseRequiredError) as exc:
            cls(report)
        assert cls.standard in str(exc.value)
        assert "license" in str(exc.value).lower()
        assert exc.value.standard == cls.standard

    def test_license_error_is_runtime_error(self, report):
        # Callers can catch the broad type without importing our class.
        with pytest.raises(RuntimeError):
            EUMachineryReg2027(report)

    def test_base_generator_also_gated(self, report):
        with pytest.raises(LicenseRequiredError):
            ComplianceGenerator(report)

    def test_error_message_points_to_activation(self, report):
        with pytest.raises(LicenseRequiredError, match="open-source"):
            ISO10218(report)
