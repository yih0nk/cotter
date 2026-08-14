"""Unit tests for the reproducibility manifest (cotter.manifest)."""

from cotter.manifest import build_manifest, dependency_versions


def test_dependency_versions_lists_the_full_tracked_set():
    versions = dependency_versions()
    # every tracked dependency appears as a key, installed or not
    for dep in ("gymnasium", "mujoco", "torch", "numpy", "scipy"):
        assert dep in versions
    # these are real runtime deps, so they resolve to a version string
    assert isinstance(versions["numpy"], str)
    assert isinstance(versions["torch"], str)


def test_missing_dependency_maps_to_none():
    # a name that is not an installed distribution is reported as None,
    # not dropped — the manifest always lists the full set.
    from cotter import manifest

    assert manifest._distribution_version("definitely-not-installed-xyz") is None


def test_build_manifest_core_fields():
    m = build_manifest(cotter_version="9.9.9")
    assert m["cotter_version"] == "9.9.9"
    assert isinstance(m["python_version"], str)
    assert isinstance(m["platform"], str)
    assert isinstance(m["dependencies"], dict)


def test_build_manifest_optional_fields():
    m = build_manifest(cotter_version="1.0.0", env_id="Env-v0", base_seed=7)
    assert m["env_id"] == "Env-v0"
    assert m["base_seed"] == 7


def test_build_manifest_omits_absent_optionals():
    m = build_manifest(cotter_version="1.0.0")
    assert "env_id" not in m
    assert "base_seed" not in m


def test_extra_is_merged_last():
    m = build_manifest(cotter_version="1.0.0", extra={"policy_sha256": "sha256:abc"})
    assert m["policy_sha256"] == "sha256:abc"
