"""Reproducibility manifest for a Cotter run.

A report is only useful as compliance evidence if the run behind it can be
reproduced. The manifest captures exactly what is needed to do that: the
tool and dependency versions, the interpreter and platform, the seed, and
(added in :func:`hash_file`) a content hash of the policy under test.

It is a plain dict so it serializes straight into the JSON report and is
rendered into the HTML page. Nothing here imports the heavy simulation
stack — only version metadata — so building a manifest is cheap.
"""

from __future__ import annotations

import platform
from importlib import metadata

# The runtime dependencies whose versions materially affect a result.
_TRACKED_DEPENDENCIES = (
    "gymnasium",
    "mujoco",
    "stable-baselines3",
    "torch",
    "numpy",
    "scipy",
    "gymnasium-robotics",
)


def _distribution_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def dependency_versions() -> dict[str, str | None]:
    """Installed versions of the dependencies that affect a result.

    A dependency that is not installed maps to ``None`` rather than being
    omitted, so the manifest always lists the full set it cares about.
    """
    return {dist: _distribution_version(dist) for dist in _TRACKED_DEPENDENCIES}


def build_manifest(
    *,
    cotter_version: str,
    env_id: str | None = None,
    base_seed: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the reproducibility manifest for a run.

    ``extra`` is merged in last, so a caller can attach run-specific fields
    (e.g. a policy hash) without this function needing to know about them.
    """
    manifest: dict = {
        "cotter_version": cotter_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
    }
    if env_id is not None:
        manifest["env_id"] = env_id
    if base_seed is not None:
        manifest["base_seed"] = base_seed
    if extra:
        manifest.update(extra)
    return manifest
