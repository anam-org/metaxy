"""Tests for feature_version() method."""

from syrupy.assertion import SnapshotAssertion

from metaxy import (
    ContainerDep,
    ContainerKey,
    ContainerSpec,
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
)


def test_feature_version_deterministic(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version is deterministic."""

    class TestFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "feature"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    version1 = TestFeature.feature_version()
    version2 = TestFeature.feature_version()

    # Should be deterministic
    assert version1 == version2

    # Should be 64 characters
    assert len(version1) == 64

    # Should be hex string
    assert all(c in "0123456789abcdef" for c in version1)

    # Snapshot the hash
    assert version1 == snapshot


def test_feature_version_changes_with_code_version(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version changes when code_version changes."""
    from metaxy.models.feature import FeatureRegistry

    # Use separate registries
    registry_v1 = FeatureRegistry()
    registry_v2 = FeatureRegistry()

    class FeatureV1(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["versioned", "feature", "test_v1"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
        registry=registry_v1,
    ):
        pass

    class FeatureV2(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["versioned", "feature", "test_v2"]),
            deps=None,
            containers=[
                ContainerSpec(
                    key=ContainerKey(["default"]), code_version=2
                ),  # Changed!
            ],
        ),
        registry=registry_v2,
    ):
        pass

    v1 = FeatureV1.feature_version()
    v2 = FeatureV2.feature_version()

    # Should be different (different code_version)
    assert v1 != v2

    # Snapshot both
    assert {"v1": v1, "v2": v2} == snapshot


def test_feature_version_changes_with_dependencies(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version changes when dependencies change."""

    class UpstreamFeature(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_deps", "upstream"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    class DownstreamNoDeps(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_deps", "downstream", "no_deps"]),
            deps=None,  # No dependencies
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    class DownstreamWithDeps(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_deps", "downstream", "with_deps"]),
            deps=[
                FeatureDep(key=FeatureKey(["test_deps", "upstream"]))
            ],  # Added dependency!
            containers=[
                ContainerSpec(key=ContainerKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    no_deps = DownstreamNoDeps.feature_version()
    with_deps = DownstreamWithDeps.feature_version()

    # Should be different
    assert no_deps != with_deps

    # Snapshot both
    assert {"no_deps": no_deps, "with_deps": with_deps} == snapshot


def test_feature_version_multi_container(snapshot: SnapshotAssertion) -> None:
    """Test feature_version with multiple containers."""

    class MultiContainer(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["multi"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
                ContainerSpec(key=ContainerKey(["audio"]), code_version=1),
                ContainerSpec(key=ContainerKey(["metadata"]), code_version=2),
            ],
        ),
    ):
        pass

    version = MultiContainer.feature_version()

    # Should be 64 characters
    assert len(version) == 64

    # Snapshot
    assert version == snapshot


def test_feature_version_with_container_deps(snapshot: SnapshotAssertion) -> None:
    """Test feature_version includes container-level dependencies."""

    class Upstream(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_container_deps", "upstream"]),
            deps=None,
            containers=[
                ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
                ContainerSpec(key=ContainerKey(["audio"]), code_version=1),
            ],
        ),
    ):
        pass

    class DownstreamNoDeps(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_container_deps", "downstream", "no_deps"]),
            deps=[FeatureDep(key=FeatureKey(["test_container_deps", "upstream"]))],
            containers=[
                ContainerSpec(
                    key=ContainerKey(["default"]),
                    code_version=1,
                    # No container deps
                ),
            ],
        ),
    ):
        pass

    class DownstreamWithContainerDeps(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test_container_deps", "downstream", "with_deps"]),
            deps=[FeatureDep(key=FeatureKey(["test_container_deps", "upstream"]))],
            containers=[
                ContainerSpec(
                    key=ContainerKey(["default"]),
                    code_version=1,
                    deps=[
                        ContainerDep(
                            feature_key=FeatureKey(["test_container_deps", "upstream"]),
                            containers=[
                                ContainerKey(["frames"]),
                                ContainerKey(["audio"]),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    no_container_deps = DownstreamNoDeps.feature_version()
    with_container_deps = DownstreamWithContainerDeps.feature_version()

    # Should be different
    assert no_container_deps != with_container_deps

    # Snapshot both
    assert {
        "no_container_deps": no_container_deps,
        "with_container_deps": with_container_deps,
    } == snapshot
