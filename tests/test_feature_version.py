"""Tests for feature_version() method."""

from syrupy.assertion import SnapshotAssertion

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    TestingFeatureSpec,
)


def test_feature_version_deterministic(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version is deterministic."""

    class TestFeature(
        Feature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version=1),
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
    from metaxy.models.feature import FeatureGraph

    # Use separate registries with context managers
    graph_v1 = FeatureGraph()
    graph_v2 = FeatureGraph()

    with graph_v1.use():

        class FeatureV1(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["versioned", "feature", "test_v1"]),
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

    with graph_v2.use():

        class FeatureV2(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["versioned", "feature", "test_v2"]),
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=2),  # Changed!
                ],
            ),
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
        spec=TestingFeatureSpec(
            key=FeatureKey(["test_deps", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    class DownstreamNoDeps(
        Feature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["test_deps", "downstream", "no_deps"]),
            # No dependencies
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version=1),
            ],
        ),
    ):
        pass

    class DownstreamWithDeps(
        Feature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["test_deps", "downstream", "with_deps"]),
            deps=[
                FeatureDep(feature=FeatureKey(["test_deps", "upstream"]))
            ],  # Added dependency!
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version=1),
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


def test_feature_version_multi_field(snapshot: SnapshotAssertion) -> None:
    """Test feature_version with multiple fields."""

    class MultiField(
        Feature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["multi"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version=1),
                FieldSpec(key=FieldKey(["audio"]), code_version=1),
                FieldSpec(key=FieldKey(["metadata"]), code_version=2),
            ],
        ),
    ):
        pass

    version = MultiField.feature_version()

    # Should be 64 characters
    assert len(version) == 64

    # Snapshot
    assert version == snapshot


def test_feature_version_with_field_deps(snapshot: SnapshotAssertion) -> None:
    """Test feature_version includes field-level dependencies."""

    class Upstream(
        Feature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["test_field_deps", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version=1),
                FieldSpec(key=FieldKey(["audio"]), code_version=1),
            ],
        ),
    ):
        pass

    class DownstreamNoDeps(
        Feature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["test_field_deps", "downstream", "no_deps"]),
            deps=[FeatureDep(feature=FeatureKey(["test_field_deps", "upstream"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["default"]),
                    code_version=1,
                    # No field deps
                ),
            ],
        ),
    ):
        pass

    class DownstreamWithFieldDeps(
        Feature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["test_field_deps", "downstream", "with_deps"]),
            deps=[FeatureDep(feature=FeatureKey(["test_field_deps", "upstream"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["default"]),
                    code_version=1,
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["test_field_deps", "upstream"]),
                            fields=[
                                FieldKey(["frames"]),
                                FieldKey(["audio"]),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    no_field_deps = DownstreamNoDeps.feature_version()
    with_field_deps = DownstreamWithFieldDeps.feature_version()

    # Should be different
    assert no_field_deps != with_field_deps

    # Snapshot both
    assert {
        "no_field_deps": no_field_deps,
        "with_field_deps": with_field_deps,
    } == snapshot
