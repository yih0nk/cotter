"""Filesystem-backed registry of trained adversary artifacts.

Layout under the zoo root (default ``~/.cotter/zoo``)::

    index.json                  # one record per entry, append-updated
    <env_id>/<victim_hash>/eps_<epsilon>/adversary.zip

Entries are keyed by ``(env_id, victim_hash, epsilon)``. The victim
hash is a SHA-256 over the victim policy — its artifact file when given
a path, or its parameter tensors when given an in-memory policy — so a
retrained victim never silently reuses a stale attacker.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from cotter.policy import Policy, SB3Policy, TorchPolicy
from cotter.tests.adversarial import PPOAdversary

DEFAULT_ROOT = Path.home() / ".cotter" / "zoo"
_HASH_LEN = 16  # hex chars; 64 bits is plenty for a local registry


def victim_hash(victim: str | Path | Policy) -> str:
    """Stable identity hash for a victim policy.

    Accepts a policy artifact path (hashes file bytes) or a loaded
    SB3Policy/TorchPolicy (hashes parameter tensors), so the same
    checkpoint hashes identically however it is referenced.
    """
    digest = hashlib.sha256()
    if isinstance(victim, (str, Path)):
        path = Path(victim)
        if not path.exists():
            raise FileNotFoundError(f"cannot hash missing policy file: {path}")
        digest.update(path.read_bytes())
    elif isinstance(victim, SB3Policy):
        for key, tensor in sorted(victim.model.policy.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.cpu().numpy().tobytes())
    elif isinstance(victim, TorchPolicy):
        for key, tensor in sorted(victim.module.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.cpu().numpy().tobytes())
    else:
        raise TypeError(
            f"cannot hash victim of type {type(victim).__name__}; pass an "
            "artifact path, SB3Policy, or TorchPolicy"
        )
    return digest.hexdigest()[:_HASH_LEN]


@dataclass(frozen=True)
class ZooEntry:
    env_id: str
    victim_hash: str
    epsilon: float
    path: str  # adversary artifact, relative to the zoo root
    algo: str
    created_at: str
    notes: str = ""

    def key(self) -> tuple[str, str, float]:
        return (self.env_id, self.victim_hash, self.epsilon)


class AdversaryZoo:
    """Save, look up, and reload adversaries keyed by what they attack."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def _read_index(self) -> list[ZooEntry]:
        if not self.index_path.exists():
            return []
        raw = json.loads(self.index_path.read_text())
        return [ZooEntry(**record) for record in raw]

    def _write_index(self, entries: list[ZooEntry]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps([asdict(e) for e in entries], indent=2) + "\n"
        )

    def entries(self, env_id: str | None = None) -> list[ZooEntry]:
        found = self._read_index()
        if env_id is not None:
            found = [e for e in found if e.env_id == env_id]
        return found

    def lookup(
        self, env_id: str, victim: str | Path | Policy, epsilon: float
    ) -> ZooEntry | None:
        key = (env_id, victim_hash(victim), epsilon)
        for entry in self._read_index():
            if entry.key() == key:
                return entry
        return None

    def save(
        self,
        adversary: PPOAdversary,
        env_id: str,
        victim: str | Path | Policy,
        notes: str = "",
    ) -> ZooEntry:
        """Store a trained adversary, replacing any entry with the same key."""
        if not isinstance(adversary, PPOAdversary):
            raise TypeError(
                f"the zoo stores trained PPOAdversary artifacts; got "
                f"{type(adversary).__name__} (the random baseline needs no storage)"
            )
        vhash = victim_hash(victim)
        rel = Path(env_id) / vhash / f"eps_{adversary.epsilon:g}" / "adversary.zip"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        adversary.model.save(target)

        entry = ZooEntry(
            env_id=env_id,
            victim_hash=vhash,
            epsilon=adversary.epsilon,
            path=str(rel),
            algo=type(adversary.model).__name__,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            notes=notes,
        )
        entries = [e for e in self._read_index() if e.key() != entry.key()]
        entries.append(entry)
        self._write_index(entries)
        return entry

    def load(
        self, env_id: str, victim: str | Path | Policy, epsilon: float
    ) -> PPOAdversary | None:
        """Reload the stored adversary for this exact (env, victim, budget)."""
        entry = self.lookup(env_id, victim, epsilon)
        if entry is None:
            return None
        from stable_baselines3 import PPO

        artifact = self.root / entry.path
        if not artifact.exists():
            raise FileNotFoundError(
                f"zoo index references {artifact} but the file is gone; "
                "the zoo directory was modified outside the registry"
            )
        return PPOAdversary(PPO.load(artifact, device="cpu"), entry.epsilon)
