"""Unit tests for the YAML config schema."""

from pathlib import Path

import pytest

from cotter.config import ConfigError, load_config, parse_config

FULL = {
    "env": "InvertedPendulum-v5",
    "algo": "PPO",
    "base_seed": 7,
    "success": {"type": "min_length", "value": 1000},
    "performance": {"p0": 0.8, "p1": 0.95, "alpha": 0.05, "beta": 0.05, "n_max": 50},
    "safety": {"n_episodes": 10, "limits": {"cotter/joint_velocities": 5.0}},
    "regression": {"baseline": "base.zip", "n_pairs": 30},
    "adversarial": {"epsilon": 0.07, "train": False},
    "report": "out/report.json",
    "report_html": "out/report.html",
    "report_junit": "out/report.xml",
    "report_md": "out/report.md",
}


class TestParseConfig:
    def test_full_config(self):
        cfg = parse_config(dict(FULL))
        assert cfg.env == "InvertedPendulum-v5"
        assert cfg.base_seed == 7
        assert cfg.performance.p1 == 0.95
        assert cfg.safety.n_episodes == 10
        assert cfg.safety.limits[0].quantity == "cotter/joint_velocities"
        assert cfg.safety.limits[0].max_abs == 5.0
        assert cfg.regression.n_pairs == 30
        assert cfg.adversarial.epsilon == 0.07
        assert cfg.adversarial.train is False
        assert cfg.adversarial.timesteps == 150_000  # default preserved
        assert cfg.report == Path("out/report.json")
        assert cfg.report_html == Path("out/report.html")
        assert cfg.report_junit == Path("out/report.xml")
        assert cfg.report_md == Path("out/report.md")
        assert cfg.success_fn()(0.0, 1000, False, True, {})

    def test_minimal_config_categories_none(self):
        cfg = parse_config({"env": "X-v1", "success": {"type": "min_return", "value": 1}})
        assert cfg.performance is None
        assert cfg.safety is None
        assert cfg.regression is None
        assert cfg.adversarial is None
        assert cfg.report is None
        assert cfg.report_html is None
        assert cfg.report_junit is None
        assert cfg.report_md is None
        assert cfg.algo == "PPO"
        assert cfg.backend == "gymnasium"  # default backend

    def test_adversarial_attack_defaults_to_observation(self):
        cfg = parse_config({
            "env": "X-v1", "success": {"type": "min_return", "value": 1},
            "adversarial": {"epsilon": 0.05},
        })
        assert cfg.adversarial.attack == "observation"

    def test_adversarial_attack_action_and_both(self):
        for surface in ("action", "both"):
            cfg = parse_config({
                "env": "X-v1", "success": {"type": "min_return", "value": 1},
                "adversarial": {"epsilon": 0.05, "attack": surface},
            })
            assert cfg.adversarial.attack == surface

    def test_invalid_attack_surface_rejected(self):
        with pytest.raises(ConfigError, match="attack"):
            parse_config({
                "env": "X-v1", "success": {"type": "min_return", "value": 1},
                "adversarial": {"epsilon": 0.05, "attack": "weights"},
            })

    def test_backend_override(self):
        cfg = parse_config({
            "env": "X-v1", "backend": "isaac-sim",
            "success": {"type": "min_return", "value": 1},
        })
        assert cfg.backend == "isaac-sim"

    def test_relative_paths_resolve_against_config_dir(self):
        cfg = parse_config(dict(FULL), config_dir=Path("/cfg/dir"))
        assert cfg.regression.baseline == Path("/cfg/dir/base.zip")
        assert cfg.report == Path("/cfg/dir/out/report.json")
        assert cfg.report_html == Path("/cfg/dir/out/report.html")

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda d: d.pop("env"), "must set 'env'"),
            (lambda d: d.pop("success"), "must set 'success'"),
            (lambda d: d.update(bogus=1), "unknown top-level"),
            (lambda d: d["performance"].update(p9=1), "unknown keys in 'performance'"),
            (lambda d: d["safety"].pop("limits"), "requires a 'limits'"),
            (lambda d: d["safety"].update(limits={}), "non-empty mapping"),
            (lambda d: d["safety"].update(limits={"q": "fast"}), "must be numeric"),
            (lambda d: d["regression"].pop("baseline"), "requires a 'baseline'"),
            (lambda d: d.update(success={"type": "nope"}), "unknown success criterion"),
            (lambda d: d.update(safety=[1, 2]), "must be a mapping"),
        ],
    )
    def test_invalid_configs_rejected(self, mutate, match):
        data = {k: dict(v) if isinstance(v, dict) else v for k, v in FULL.items()}
        mutate(data)
        with pytest.raises((ConfigError, ValueError), match=match):
            parse_config(data)

    def test_negative_safety_limit_rejected(self):
        data = dict(FULL, safety={"limits": {"q": -1.0}})
        with pytest.raises(ConfigError):
            parse_config(data)


class TestLoadConfig:
    def test_yaml_roundtrip(self, tmp_path):
        cfg_file = tmp_path / "run.yaml"
        cfg_file.write_text(
            "env: InvertedPendulum-v5\n"
            "success: {type: min_length, value: 1000}\n"
            "safety:\n"
            "  limits:\n"
            "    cotter/actuator_forces: 2.5\n"
            "regression: {baseline: base.zip}\n"
        )
        cfg = load_config(cfg_file)
        assert cfg.safety.limits[0].max_abs == 2.5
        assert cfg.regression.baseline == tmp_path / "base.zip"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("no_such.yaml")

    def test_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("env: [unclosed")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(bad)

    def test_error_names_file(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("success: {type: min_length, value: 1}\n")
        with pytest.raises(ConfigError, match="bad.yaml"):
            load_config(bad)
