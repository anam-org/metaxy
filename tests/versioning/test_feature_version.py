"""Tests for feature_version() method."""

from metaxy_testing.models import SampleFeatureSpec
from syrupy.assertion import SnapshotAssertion

from metaxy import BaseFeature, FeatureDep, FeatureKey, FieldDep, FieldKey, FieldSpec


def test_feature_version_deterministic(snapshot: SnapshotAssertion) -> None:
    """Test that feature_version is deterministic."""

    class TestFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    version1 = TestFeature.feature_version()
    version2 = TestFeature.feature_version()

    # Should be deterministic
    assert version1 == version2

    # Should be 8 characters
    assert len(version1) == 8

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
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["versioned", "feature", "test_v1"]),
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="1"),
                ],
            ),
        ):
            pass

    with graph_v2.use():

        class FeatureV2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["versioned", "feature", "test_v2"]),
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version="2"),  # Changed!
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
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test_deps", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DownstreamNoDeps(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test_deps", "downstream", "no_deps"]),
            # No dependencies
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DownstreamWithDeps(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test_deps", "downstream", "with_deps"]),
            deps=[FeatureDep(feature=FeatureKey(["test_deps", "upstream"]))],  # Added dependency!
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
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
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["multi"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["metadata"]), code_version="2"),
            ],
        ),
    ):
        pass

    version = MultiField.feature_version()

    # Should be 8 characters
    assert len(version) == 8

    # Snapshot
    assert version == snapshot


def test_feature_version_with_field_deps(snapshot: SnapshotAssertion) -> None:
    """Test feature_version includes field-level dependencies."""

    class Upstream(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test_field_deps", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    class DownstreamNoDeps(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test_field_deps", "downstream", "no_deps"]),
            deps=[FeatureDep(feature=FeatureKey(["test_field_deps", "upstream"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["default"]),
                    code_version="1",
                    # No field deps
                ),
            ],
        ),
    ):
        pass

    class DownstreamWithFieldDeps(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test_field_deps", "downstream", "with_deps"]),
            deps=[FeatureDep(feature=FeatureKey(["test_field_deps", "upstream"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["default"]),
                    code_version="1",
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


def test_external_feature_version_matches_normal_feature() -> None:
    """Test that external features produce the same version as normal features.

    The feature version is computed from field keys and code_versions only.
    Other properties like the Pydantic schema should not affect the version.
    """
    from metaxy.models.feature import FeatureDefinition, FeatureGraph

    graph = FeatureGraph()
    with graph.use():
        # Define a normal class-based feature
        class NormalFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_test", "normal"]),
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version="1"),
                    FieldSpec(key=FieldKey(["metadata"]), code_version="2"),
                ],
            ),
        ):
            pass

        # Create an external feature with matching keys and code_versions
        external_def = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["version_test", "external"]),
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version="1"),
                    FieldSpec(key=FieldKey(["metadata"]), code_version="2"),
                ],
            ),
            # Different schema - should NOT affect version
            feature_schema={"type": "object", "properties": {"different": {"type": "string"}}},
            project="external-project",
        )
        graph.add_feature_definition(external_def)

        # Get versions - they should match because fields and code_versions match
        # (feature key is different, so the version will be different, but the
        # computation logic should be identical)
        normal_version = graph.get_feature_version(["version_test", "normal"])
        external_version = graph.get_feature_version(["version_test", "external"])

        # Versions are different because feature keys are different
        # (the key is part of the hash)
        assert normal_version != external_version

        # But if we create features with the SAME key, versions should match
        graph2 = FeatureGraph()
        with graph2.use():

            class SameKeyNormal(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["version_test", "same_key"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
            ):
                pass

            normal_version_same_key = graph2.get_feature_version(["version_test", "same_key"])

        graph3 = FeatureGraph()
        with graph3.use():
            external_same_key = FeatureDefinition.external(
                spec=SampleFeatureSpec(
                    key=FeatureKey(["version_test", "same_key"]),
                    fields=[
                        FieldSpec(key=FieldKey(["value"]), code_version="1"),
                    ],
                ),
                # Completely different schema
                feature_schema={"totally": "different", "schema": True},
                project="another-project",
            )
            graph3.add_feature_definition(external_same_key)

            external_version_same_key = graph3.get_feature_version(["version_test", "same_key"])

        # Same key, same fields, same code_versions -> same version
        assert normal_version_same_key == external_version_same_key


def test_external_feature_version_with_dependencies() -> None:
    """Test that external features used as dependencies produce correct downstream versions.

    When a downstream feature depends on an external feature, the version computation
    should work the same as if it depended on a normal feature with the same spec.
    """
    from metaxy.models.feature import FeatureDefinition, FeatureGraph

    # Graph 1: Normal upstream + downstream
    graph1 = FeatureGraph()
    with graph1.use():

        class NormalUpstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["dep_test", "upstream"]),
                fields=[
                    FieldSpec(key=FieldKey(["source_data"]), code_version="1"),
                ],
            ),
        ):
            pass

        class NormalDownstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["dep_test", "downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["dep_test", "upstream"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["derived"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["dep_test", "upstream"]),
                                fields=[FieldKey(["source_data"])],
                            )
                        ],
                    ),
                ],
            ),
        ):
            pass

        normal_downstream_version = graph1.get_feature_version(["dep_test", "downstream"])

    # Graph 2: External upstream + same downstream
    graph2 = FeatureGraph()
    with graph2.use():
        # Add upstream as external feature (same spec)
        external_upstream = FeatureDefinition.external(
            spec=SampleFeatureSpec(
                key=FeatureKey(["dep_test", "upstream"]),
                fields=[
                    FieldSpec(key=FieldKey(["source_data"]), code_version="1"),
                ],
            ),
            feature_schema={"different": "schema"},
            project="external-project",
        )
        graph2.add_feature_definition(external_upstream)

        # Define same downstream (must use same spec)
        class ExternalDownstream(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["dep_test", "downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["dep_test", "upstream"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["derived"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["dep_test", "upstream"]),
                                fields=[FieldKey(["source_data"])],
                            )
                        ],
                    ),
                ],
            ),
        ):
            pass

        external_downstream_version = graph2.get_feature_version(["dep_test", "downstream"])

    # Downstream versions should match regardless of whether upstream is external
    assert normal_downstream_version == external_downstream_version
