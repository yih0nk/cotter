"""Adversary zoo: a registry of trained adversary artifacts.

Adversaries are expensive to train and specific to what they attack, so
the zoo keys each artifact by ``(env_id, victim_hash, epsilon)`` — the
same victim on the same env at the same budget reuses the stored
attacker instead of retraining. The open-source engine trains and
stores your own adversaries here; a hosted zoo of pretrained experts is
the planned paid tier.
"""

from cotter.zoo.registry import AdversaryZoo, ZooEntry, victim_hash

__all__ = ["AdversaryZoo", "ZooEntry", "victim_hash"]
