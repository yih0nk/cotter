"""Tests for clause traceability (cotter.traceability)."""

from dataclasses import dataclass

from cotter.traceability import build_traceability


@dataclass
class R:
    category: str
    name: str
    passed: bool | None


MAPPING = {
    "EHSR-1.3.7": {"description": "moving parts", "checks": ["hard_limits"]},
    "ISO-15066": {"description": "PFL", "checks": ["safety/iso_ts_15066_pfl"]},
    "AI-Act-15": {"description": "robustness", "checks": ["learned_ppo"]},
    "uncovered": {"description": "nothing maps here", "checks": ["missing_check"]},
}


class TestBuildTraceability:
    def test_verified_failed_unverified(self):
        results = [
            R("safety", "hard_limits", True),          # -> EHSR-1.3.7 verified
            R("safety", "iso_ts_15066_pfl", True),     # -> ISO-15066 verified
            R("adversarial", "learned_ppo", False),    # -> AI-Act-15 failed
        ]
        t = build_traceability(results, MAPPING)
        by = {c.clause: c.status for c in t.clauses}
        assert by["EHSR-1.3.7"] == "verified"
        assert by["ISO-15066"] == "verified"
        assert by["AI-Act-15"] == "failed"
        assert by["uncovered"] == "unverified"

    def test_all_verified_gate(self):
        results = [R("safety", "hard_limits", True)]
        t = build_traceability(results, {"c": {"description": "", "checks": ["hard_limits"]}})
        assert t.all_verified

    def test_gaps_when_unverified(self):
        t = build_traceability([], MAPPING)
        assert not t.all_verified
        assert len(t.unverified) == 4

    def test_category_qualified_match(self):
        results = [R("safety", "iso_ts_15066_pfl", True)]
        t = build_traceability(results, {"c": {"checks": ["safety/iso_ts_15066_pfl"]}})
        assert t.clauses[0].status == "verified"
        assert t.clauses[0].evidence == ["safety/iso_ts_15066_pfl"]

    def test_informational_alone_is_unverified(self):
        results = [R("performance", "sprt", None)]
        t = build_traceability(results, {"c": {"checks": ["sprt"]}})
        assert t.clauses[0].status == "unverified"

    def test_summary_and_dict(self):
        t = build_traceability([R("safety", "hard_limits", True)], MAPPING)
        assert "GAPS" in t.summary()  # AI-Act-15 + others unverified
        assert t.to_dict()["n_verified"] == 1
