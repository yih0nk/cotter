"""Declarative success criteria for config-driven runs.

The runner takes an arbitrary Python predicate; config files need a
small declarative vocabulary instead. Three criterion types cover the
common cases:

* ``min_length`` — episode survived at least N steps (stabilization
  tasks like InvertedPendulum).
* ``min_return`` — total episode reward reached a threshold.
* ``info_flag`` — a truthy key in the final step's info dict
  (goal-conditioned envs: gymnasium-robotics sets ``is_success``).
"""

from __future__ import annotations

from typing import Mapping

from cotter.runner import SuccessFn

CRITERION_TYPES = ("min_length", "min_return", "info_flag", "min_info")


def make_success_fn(spec: Mapping) -> SuccessFn:
    """Build a success predicate from a config mapping.

    Examples::

        make_success_fn({"type": "min_length", "value": 1000})
        make_success_fn({"type": "min_return", "value": 950.0})
        make_success_fn({"type": "info_flag", "key": "is_success"})
        make_success_fn({"type": "min_info", "key": "x_position", "value": 1.0})

    ``min_info`` succeeds when a numeric final-step info value reaches a
    threshold — e.g. forward displacement for a locomotion policy, which
    is the task metric rather than raw episode reward.
    """
    if "type" not in spec:
        raise ValueError(
            f"success criterion needs a 'type' (one of {CRITERION_TYPES}); got {dict(spec)}"
        )
    kind = spec["type"]

    if kind == "min_length":
        threshold = _require_number(spec, "value", kind)

        def success_fn(total_reward, length, terminated, truncated, final_info):
            return length >= threshold

    elif kind == "min_return":
        threshold = _require_number(spec, "value", kind)

        def success_fn(total_reward, length, terminated, truncated, final_info):
            return total_reward >= threshold

    elif kind == "info_flag":
        key = spec.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"success criterion 'info_flag' needs a string 'key'; got {dict(spec)}")

        def success_fn(total_reward, length, terminated, truncated, final_info):
            if key not in final_info:
                raise KeyError(
                    f"success criterion reads info['{key}'] but the final step info "
                    f"only contains {sorted(final_info.keys())}"
                )
            return bool(final_info[key])

    elif kind == "min_info":
        key = spec.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"success criterion 'min_info' needs a string 'key'; got {dict(spec)}")
        threshold = _require_number(spec, "value", kind)

        def success_fn(total_reward, length, terminated, truncated, final_info):
            if key not in final_info:
                raise KeyError(
                    f"success criterion reads info['{key}'] but the final step info "
                    f"only contains {sorted(final_info.keys())}"
                )
            return float(final_info[key]) >= threshold

    else:
        raise ValueError(f"unknown success criterion type '{kind}' (expected one of {CRITERION_TYPES})")

    return success_fn


def _require_number(spec: Mapping, field: str, kind: str) -> float:
    value = spec.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"success criterion '{kind}' needs a numeric '{field}'; got {dict(spec)}")
    return float(value)
