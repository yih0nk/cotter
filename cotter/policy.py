"""Black-box policy loading and observation/action-space validation.

Cotter treats a policy as a black box mapping observation -> action.
Three concrete backends are supported:

* Stable-Baselines3 models (a loaded ``BaseAlgorithm`` or a ``.zip`` path)
* Raw PyTorch modules (an ``nn.Module`` instance or a ``.pt`` path saved
  with ``torch.save``/``torch.jit.save``)
* ONNX models (a ``.onnx`` path). Requires ``onnxruntime`` — install the
  optional extra with ``pip install cotterbot[onnx]``.

Every load path runs :func:`validate_spaces`, which checks the policy's
expected observation/action shapes against the target environment and
fails loudly on mismatch — the most common real-world integration bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm


class SpaceMismatchError(ValueError):
    """The policy's expected spaces do not match the environment's."""


@runtime_checkable
class Policy(Protocol):
    """Anything with a deterministic ``predict(obs) -> action``."""

    name: str

    def predict(self, obs: np.ndarray) -> np.ndarray: ...


class SB3Policy:
    """Wrap a Stable-Baselines3 model as a deterministic black-box policy."""

    def __init__(self, model: BaseAlgorithm, name: str | None = None) -> None:
        self.model = model
        self.name = name or type(model).__name__

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action)


class TorchPolicy:
    """Wrap a raw ``torch.nn.Module`` (obs tensor in, action tensor out)."""

    def __init__(self, module: torch.nn.Module, name: str | None = None) -> None:
        self.module = module.eval()
        self.name = name or type(module).__name__

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        action = self.module(tensor)
        return action.detach().cpu().numpy()


class OnnxPolicy:
    """Wrap an ONNX model as a deterministic black-box policy.

    Supports single-input (Box observation) models — the common export
    shape ``[batch, obs_dim] -> [batch, act_dim]``. A leading batch axis
    is added/stripped automatically by comparing the model's declared
    input rank against the observation's rank, so a single observation
    round-trips whether or not the export fixed a batch dimension.
    """

    def __init__(self, session, name: str, input_name: str, input_rank: int) -> None:
        self.session = session
        self.name = name
        self._input_name = input_name
        self._input_rank = input_rank

    @classmethod
    def from_path(cls, path: str | Path, name: str | None = None) -> "OnnxPolicy":
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised via load_policy
            raise ImportError(
                "loading an ONNX policy requires onnxruntime; install the "
                "optional extra with 'pip install cotterbot[onnx]'"
            ) from exc

        path = Path(path)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise SpaceMismatchError(
                f"ONNX policy '{path.name}' has {len(inputs)} inputs; only "
                "single-input (Box observation) models are supported. Dict "
                "observation models are not yet handled."
            )
        spec = inputs[0]
        return cls(session, name or path.stem, spec.name, len(spec.shape))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        add_batch = self._input_rank == arr.ndim + 1
        payload = arr[None, ...] if add_batch else arr
        output = np.asarray(self.session.run(None, {self._input_name: payload})[0])
        return output[0] if add_batch else output


def _obs_shape_signature(space: gym.Space):
    """Comparable shape description for Box and Dict observation spaces."""
    if isinstance(space, gym.spaces.Dict):
        return {key: sub.shape for key, sub in space.spaces.items()}
    return space.shape


def validate_spaces(policy: Policy, env: gym.Env) -> None:
    """Check the policy against the env's observation/action spaces.

    For SB3 models the declared spaces are compared shape-for-shape —
    per key for Dict observation spaces (gymnasium-robotics style). For
    every policy a functional probe is run: predict on a sampled
    observation and verify the returned action's shape and finiteness.
    Raises :class:`SpaceMismatchError` with an actionable message.
    """
    obs_space, act_space = env.observation_space, env.action_space

    if isinstance(policy, SB3Policy):
        model = policy.model
        expected = _obs_shape_signature(model.observation_space)
        actual = _obs_shape_signature(obs_space)
        if expected != actual:
            raise SpaceMismatchError(
                f"policy '{policy.name}' was trained on observations of shape "
                f"{expected} but the environment produces {actual}. You are "
                "almost certainly loading the wrong checkpoint or wrapping the "
                "env differently than at training time."
            )
        if model.action_space.shape != act_space.shape:
            raise SpaceMismatchError(
                f"policy '{policy.name}' emits actions of shape "
                f"{model.action_space.shape} but the environment expects "
                f"{act_space.shape}."
            )

    sample_obs = obs_space.sample()
    try:
        action = np.asarray(policy.predict(sample_obs))
    except Exception as exc:
        raise SpaceMismatchError(
            f"policy '{policy.name}' failed on a sample observation of shape "
            f"{_obs_shape_signature(obs_space)}: {exc!r}. The policy input "
            "layer likely does not match the environment's observation space."
        ) from exc

    if action.shape != act_space.shape:
        raise SpaceMismatchError(
            f"policy '{policy.name}' returned an action of shape {action.shape} "
            f"but the environment expects {act_space.shape}."
        )
    if not np.all(np.isfinite(action)):
        raise SpaceMismatchError(
            f"policy '{policy.name}' returned a non-finite action: {action}"
        )


def load_policy(
    source: str | Path | BaseAlgorithm | torch.nn.Module,
    env: gym.Env,
    algo: type[BaseAlgorithm] | None = None,
    name: str | None = None,
) -> Policy:
    """Load a policy from a model object or file and validate it against ``env``.

    ``source`` may be an SB3 model, an SB3 ``.zip`` path (requires ``algo``,
    e.g. ``PPO``), a ``torch.nn.Module``, or a ``.pt`` path (tries
    ``torch.jit.load`` first, then ``torch.load``).
    """
    policy: Policy
    if isinstance(source, BaseAlgorithm):
        policy = SB3Policy(source, name=name)
    elif isinstance(source, torch.nn.Module):
        policy = TorchPolicy(source, name=name)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"policy file not found: {path}")
        if path.suffix == ".zip":
            if algo is None:
                raise ValueError(
                    "loading an SB3 .zip requires the algorithm class, e.g. "
                    "load_policy(path, env, algo=PPO)"
                )
            # Pass the env: models trained with HerReplayBuffer (SAC+HER)
            # require it at load time; harmless for other algorithms.
            model = algo.load(path, env=env, device="cpu")
            policy = SB3Policy(model, name=name or path.stem)
        elif path.suffix in (".pt", ".pth"):
            try:
                module = torch.jit.load(path, map_location="cpu")
            except RuntimeError:
                module = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(module, torch.nn.Module):
                raise TypeError(
                    f"{path} did not contain an nn.Module (got {type(module).__name__}); "
                    "save the full module, not just a state_dict"
                )
            policy = TorchPolicy(module, name=name or path.stem)
        elif path.suffix == ".onnx":
            policy = OnnxPolicy.from_path(path, name=name)
        else:
            raise ValueError(
                f"unsupported policy file type '{path.suffix}' "
                "(expected .zip, .pt/.pth, or .onnx)"
            )

    validate_spaces(policy, env)
    return policy
