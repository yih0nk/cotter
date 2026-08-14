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

import hashlib
import platform
from importlib import metadata
from pathlib import Path

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


def hash_file(path: str | Path, *, algo: str = "sha256", chunk_size: int = 65536) -> str | None:
    """Content hash of a file, as ``"<algo>:<hexdigest>"``.

    Returns ``None`` if the path is not a readable file, so a manifest can
    record "no policy hash available" without the caller special-casing it.
    Streamed in chunks so hashing a large policy artifact stays cheap on
    memory.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.new(algo)
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return f"{algo}:{digest.hexdigest()}"


def build_manifest(
    *,
    cotter_version: str,
    env_id: str | None = None,
    base_seed: int | None = None,
    policy_path: str | Path | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the reproducibility manifest for a run.

    When ``policy_path`` is given, its content hash is recorded so a report
    can be tied to the exact policy artifact it tested. ``extra`` is merged
    in last for any additional run-specific fields.
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
    if policy_path is not None:
        manifest["policy_path"] = str(policy_path)
        manifest["policy_sha256"] = hash_file(policy_path)
    if extra:
        manifest.update(extra)
    return manifest
