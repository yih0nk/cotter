"""Class-keyed registry of pretrained adversary artifacts.

Unlike :class:`~cotter.zoo.registry.AdversaryZoo` — which caches an
adversary for one *specific* victim — the pretrained zoo is a curated
library keyed by **robot class**: an adversary trained per class (a
reference victim or an ensemble on that env) transfers to *any*
compatible victim on the same environment, so a user can attack their
own policy without training one. Adversarial policies transfer because
they exploit the environment/observation structure, not victim-specific
weights (Gleave et al. 2019).

The open engine ships the mechanism (register / list / load / apply); a
hosted library of GPU-trained experts per robot class is the paid tier.

Layout under the root (default ``~/.cotter/pretrained``)::

    index.json
    <robot_class>/<env_id>/<attack>/eps_<epsilon>/adversary.zip

Entries are keyed by ``(robot_class, env_id, epsilon, attack)`` where
``attack`` is ``"observation"`` or ``"action"``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cotter" / "pretrained"
_ATTACKS = ("observation", "action")


@dataclass(frozen=True)
class PretrainedEntry:
    robot_class: str
    env_id: str
    epsilon: float
    attack: str  # observation | action
    path: str  # adversary artifact, relative to the zoo root
    algo: str
    created_at: str
    notes: str = ""

    def key(self) -> tuple[str, str, float, str]:
        return (self.robot_class, self.env_id, self.epsilon, self.attack)


class PretrainedZoo:
    """Register, look up, and reload pretrained adversaries by robot class."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def _read_index(self) -> list[PretrainedEntry]:
        if not self.index_path.exists():
            return []
        return [PretrainedEntry(**record) for record in json.loads(self.index_path.read_text())]

    def _write_index(self, entries: list[PretrainedEntry]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps([asdict(e) for e in entries], indent=2) + "\n")

    def entries(self, robot_class: str | None = None) -> list[PretrainedEntry]:
        found = self._read_index()
        if robot_class is not None:
            found = [e for e in found if e.robot_class == robot_class]
        return found

    def lookup(
        self, robot_class: str, env_id: str, epsilon: float, attack: str = "observation"
    ) -> PretrainedEntry | None:
        key = (robot_class, env_id, epsilon, attack)
        for entry in self._read_index():
            if entry.key() == key:
                return entry
        return None

    def prune(self) -> list[PretrainedEntry]:
        """Drop index entries whose artifact files are missing; return them."""
        kept, removed = [], []
        for entry in self._read_index():
            (kept if (self.root / entry.path).exists() else removed).append(entry)
        if removed:
            self._write_index(kept)
        return removed

    def register(
        self,
        adversary,
        robot_class: str,
        env_id: str,
        attack: str = "observation",
        notes: str = "",
    ) -> PretrainedEntry:
        """Store a trained adversary under a robot-class label.

        ``adversary`` is a trained ``PPOAdversary`` (observation attack) or
        ``PPOActionAdversary`` (action attack); ``attack`` must match.
        Replaces any entry with the same key.
        """
        if attack not in _ATTACKS:
            raise ValueError(f"attack must be one of {list(_ATTACKS)}; got {attack!r}")
        model = getattr(adversary, "model", None)
        epsilon = getattr(adversary, "epsilon", None)
        if model is None or epsilon is None:
            raise TypeError(
                "register expects a trained PPOAdversary/PPOActionAdversary "
                f"(with .model and .epsilon); got {type(adversary).__name__}"
            )

        rel = Path(robot_class) / env_id / attack / f"eps_{epsilon:g}" / "adversary.zip"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(target)

        entry = PretrainedEntry(
            robot_class=robot_class,
            env_id=env_id,
            epsilon=epsilon,
            attack=attack,
            path=str(rel),
            algo=type(model).__name__,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            notes=notes,
        )
        entries = [e for e in self._read_index() if e.key() != entry.key()]
        entries.append(entry)
        self._write_index(entries)
        return entry

    def load(
        self, robot_class: str, env_id: str, epsilon: float, attack: str = "observation"
    ):
        """Reload a pretrained adversary, wrapped for its attack surface."""
        entry = self.lookup(robot_class, env_id, epsilon, attack)
        if entry is None:
            return None
        from stable_baselines3 import PPO

        artifact = self.root / entry.path
        if not artifact.exists():
            raise FileNotFoundError(
                f"pretrained index references {artifact} but the file is gone; "
                "the zoo directory was modified outside the registry"
            )
        model = PPO.load(artifact, device="cpu")
        if entry.attack == "action":
            from cotter.tests.action_adversarial import PPOActionAdversary

            return PPOActionAdversary(model, entry.epsilon)
        from cotter.tests.adversarial import PPOAdversary

        return PPOAdversary(model, entry.epsilon)
