"""Transfer attacks: apply a pretrained adversary to a fresh victim.

A pretrained adversary (from :class:`~cotter.zoo.pretrained.PretrainedZoo`)
was trained against a *reference* victim on some environment. Because
adversarial policies exploit the environment/observation structure rather
than victim-specific weights, the same adversary transfers to any victim
on a compatible env. :func:`transfer_attack` validates that compatibility
and evaluates the victim under the pretrained attack.
"""

from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym

from cotter.policy import Policy
from cotter.runner import SuccessFn
from cotter.tests.adversarial import (
    AdversarialResult,
    perturbable_shape,
    run_adversarial_test,
)
from cotter.tests.action_adversarial import (
    require_box_action,
    run_action_adversarial_test,
)


class IncompatibleAdversaryError(ValueError):
    """A pretrained adversary does not match the target environment."""


def _obs_shape(space: gym.Space):
    if isinstance(space, gym.spaces.Dict):
        return {k: tuple(v.shape) for k, v in space.spaces.items()}
    return tuple(space.shape)


def check_adversary_compatible(adversary, env: gym.Env, attack: str = "observation") -> None:
    """Raise if a pretrained adversary cannot attack ``env``.

    The adversary's model must observe the env's observation space and
    emit a perturbation of the right shape for the attack surface.
    """
    model = getattr(adversary, "model", None)
    if model is None:
        raise IncompatibleAdversaryError(
            f"{type(adversary).__name__} has no trained model; only PPO-based "
            "pretrained adversaries can be transferred"
        )

    if _obs_shape(model.observation_space) != _obs_shape(env.observation_space):
        raise IncompatibleAdversaryError(
            f"pretrained adversary observes {_obs_shape(model.observation_space)} but "
            f"the env produces {_obs_shape(env.observation_space)} — different robot class"
        )

    if attack == "action":
        expected = tuple(require_box_action(env.action_space))
    else:
        expected = tuple(perturbable_shape(env.observation_space))
    if tuple(model.action_space.shape) != expected:
        raise IncompatibleAdversaryError(
            f"pretrained {attack} adversary emits perturbations of shape "
            f"{tuple(model.action_space.shape)} but this env needs {expected}"
        )


def transfer_attack(
    adversary,
    victim: Policy,
    env: gym.Env,
    success_fn: SuccessFn,
    attack: str = "observation",
    epsilon: float | None = None,
    n_episodes: int = 20,
    min_success_rate: float = 0.5,
    base_seed: int = 0,
    seeds: Sequence[int] | None = None,
    robot_class: str | None = None,
    n_workers: int = 1,
    env_factory: Callable[[], gym.Env] | None = None,
) -> AdversarialResult:
    """Evaluate ``victim`` under a pretrained adversary it never trained on.

    ``epsilon`` defaults to the adversary's own budget. Raises
    :class:`IncompatibleAdversaryError` if the adversary does not match the
    env (wrong robot class / attack surface).
    """
    check_adversary_compatible(adversary, env, attack)
    eps = epsilon if epsilon is not None else adversary.epsilon
    note = "pretrained transfer attack"
    if robot_class:
        note += f" (class={robot_class})"

    common = dict(
        n_episodes=n_episodes, adversary=adversary, min_success_rate=min_success_rate,
        base_seed=base_seed, seeds=seeds, notes=note, n_workers=n_workers,
        env_factory=env_factory,
    )
    if attack == "action":
        return run_action_adversarial_test(victim, env, success_fn, epsilon=eps, **common)
    return run_adversarial_test(victim, env, success_fn, epsilon=eps, **common)
