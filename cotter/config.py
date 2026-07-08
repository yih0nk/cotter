"""YAML config schema for CLI-driven test runs.

A config file declares which test categories to run and their
parameters; categories are optional and only the ones present execute.
Example::

    env: InvertedPendulum-v5
    algo: PPO
    base_seed: 0
    success:
      type: min_length
      value: 1000
    performance:
      p0: 0.80
      p1: 0.95
      alpha: 0.05
      beta: 0.05
      n_max: 50
    safety:
      n_episodes: 20
      limits:
        cotter/joint_velocities: 5.0
        cotter/actuator_forces: 2.5
    regression:
      baseline: artifacts/victim_ppo_inverted_pendulum.zip
      n_pairs: 30
      alpha: 0.05
    adversarial:
      epsilon: 0.07
      n_episodes: 20
      min_success_rate: 0.5
      train: true
      timesteps: 150000
    report: report.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cotter.success import make_success_fn
from cotter.tests.safety import SafetyLimit


class ConfigError(ValueError):
    """The config file is invalid; the message says where and why."""


@dataclass
class PerformanceConfig:
    p0: float = 0.80
    p1: float = 0.95
    alpha: float = 0.05
    beta: float = 0.05
    n_max: int = 50


@dataclass
class SafetyConfig:
    limits: list[SafetyLimit] = field(default_factory=list)
    n_episodes: int = 20
    n_workers: int = 1  # parallel rollout workers; 1 = serial


@dataclass
class RegressionConfig:
    baseline: Path = Path()
    n_pairs: int = 30
    alpha: float = 0.05
    n_workers: int = 1  # parallel rollout workers; 1 = serial


@dataclass
class AdversarialConfig:
    epsilon: float = 0.05
    n_episodes: int = 20
    min_success_rate: float = 0.5
    train: bool = True
    timesteps: int = 150_000
    use_zoo: bool = False  # reuse/cache the trained adversary in the zoo
    zoo_root: str | None = None  # override the default ~/.cotter/zoo
    n_workers: int = 1  # parallel rollout workers for the fixed-N eval; 1 = serial
    max_seconds: float | None = None  # wall-clock time-box on adversary training


@dataclass
class RunConfig:
    env: str
    success: dict
    algo: str = "PPO"
    base_seed: int = 0
    backend: str = "gymnasium"
    performance: PerformanceConfig | None = None
    safety: SafetyConfig | None = None
    regression: RegressionConfig | None = None
    adversarial: AdversarialConfig | None = None
    report: Path | None = None

    def success_fn(self):
        return make_success_fn(self.success)


_KNOWN_TOP_KEYS = {
    "env", "algo", "base_seed", "backend", "success",
    "performance", "safety", "regression", "adversarial", "report",
}


def _section(data: dict, name: str, cls, **transforms):
    raw = data.get(name)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"'{name}' must be a mapping; got {type(raw).__name__}")
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"unknown keys in '{name}': {sorted(unknown)} (expected {sorted(valid)})")
    kwargs = dict(raw)
    for key, fn in transforms.items():
        if key in kwargs:
            kwargs[key] = fn(kwargs[key])
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid '{name}' section: {exc}") from exc


def _parse_limits(raw) -> list[SafetyLimit]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("'safety.limits' must be a non-empty mapping of quantity -> max_abs")
    limits = []
    for quantity, max_abs in raw.items():
        if not isinstance(max_abs, (int, float)) or isinstance(max_abs, bool):
            raise ConfigError(f"safety limit '{quantity}' must be numeric; got {max_abs!r}")
        try:
            limits.append(SafetyLimit(str(quantity), float(max_abs)))
        except ValueError as exc:
            raise ConfigError(f"safety limit '{quantity}': {exc}") from exc
    return limits


def parse_config(data: dict, config_dir: Path | None = None) -> RunConfig:
    """Validate a parsed YAML mapping into a RunConfig.

    Relative paths (regression baseline, report) resolve against
    ``config_dir`` when given, so configs work from any cwd.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping; got {type(data).__name__}")
    unknown = set(data) - _KNOWN_TOP_KEYS
    if unknown:
        raise ConfigError(f"unknown top-level keys: {sorted(unknown)} (expected {sorted(_KNOWN_TOP_KEYS)})")
    if "env" not in data:
        raise ConfigError("config must set 'env' (a Gymnasium env id)")
    if "success" not in data:
        raise ConfigError("config must set 'success' (a success criterion mapping)")

    def resolve(p) -> Path:
        path = Path(p)
        if config_dir is not None and not path.is_absolute():
            path = config_dir / path
        return path

    cfg = RunConfig(
        env=str(data["env"]),
        success=dict(data["success"]),
        algo=str(data.get("algo", "PPO")),
        base_seed=int(data.get("base_seed", 0)),
        backend=str(data.get("backend", "gymnasium")),
        performance=_section(data, "performance", PerformanceConfig),
        safety=_section(data, "safety", SafetyConfig, limits=_parse_limits),
        regression=_section(data, "regression", RegressionConfig, baseline=resolve),
        adversarial=_section(data, "adversarial", AdversarialConfig),
        report=resolve(data["report"]) if "report" in data else None,
    )
    if cfg.safety is not None and not cfg.safety.limits:
        raise ConfigError("'safety' section requires a 'limits' mapping")
    if cfg.regression is not None and cfg.regression.baseline == Path():
        raise ConfigError("'regression' section requires a 'baseline' policy path")
    make_success_fn(cfg.success)  # fail at load time, not mid-run
    return cfg


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return parse_config(data, config_dir=path.resolve().parent)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
