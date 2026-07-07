"""Unit tests for declarative success criteria."""

import pytest

from cotter.success import make_success_fn


class TestMinLength:
    def test_threshold(self):
        fn = make_success_fn({"type": "min_length", "value": 1000})
        assert fn(0.0, 1000, False, True, {})
        assert not fn(0.0, 999, True, False, {})


class TestMinReturn:
    def test_threshold(self):
        fn = make_success_fn({"type": "min_return", "value": 950.0})
        assert fn(950.0, 10, False, True, {})
        assert not fn(949.9, 10, False, True, {})


class TestInfoFlag:
    def test_truthy_values(self):
        fn = make_success_fn({"type": "info_flag", "key": "is_success"})
        assert fn(0.0, 5, True, False, {"is_success": True})
        assert fn(0.0, 5, True, False, {"is_success": 1.0})  # Fetch uses float
        assert not fn(0.0, 5, True, False, {"is_success": 0.0})

    def test_missing_key_raises(self):
        fn = make_success_fn({"type": "info_flag", "key": "is_success"})
        with pytest.raises(KeyError, match="is_success"):
            fn(0.0, 5, True, False, {"other": 1})


class TestValidation:
    @pytest.mark.parametrize(
        "spec",
        [
            {},
            {"type": "nope"},
            {"type": "min_length"},  # missing value
            {"type": "min_length", "value": "1000"},  # non-numeric
            {"type": "min_length", "value": True},  # bool is not a threshold
            {"type": "info_flag"},  # missing key
            {"type": "info_flag", "key": ""},
        ],
    )
    def test_bad_specs_rejected(self, spec):
        with pytest.raises((ValueError, KeyError)):
            make_success_fn(spec)
