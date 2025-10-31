"""Tests comparing feature_spec_version and feature_version behavior."""

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldKey,
    FieldSpec,
    TestingFeatureSpec,
)


def test_feature_spec_version_vs_feature_version() -> None:
    """Test that feature_spec_version and feature_version serve different purposes.

    feature_spec_version: Hashes ALL properties of the specification (complete audit trail)
    feature_version: Only hashes computational properties (for migration triggering)
    """
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["test", "comparison"]),
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        # Get both versions
        feature_spec_version = TestFeature.spec().feature_spec_version
        feature_version = TestFeature.feature_version()

        # Both should be valid SHA256 hashes
        assert len(feature_spec_version) == 64
        assert len(feature_version) == 64
        assert all(c in "0123456789abcdef" for c in feature_spec_version)
        assert all(c in "0123456789abcdef" for c in feature_version)

        # They should be different because they hash different things
        # feature_version goes through the graph and computes based on dependencies
        # feature_spec_version directly serializes the spec to JSON
        assert feature_spec_version != feature_version


def test_feature_spec_version_stability_with_future_metadata() -> None:
    """Test that feature_spec_version will capture future metadata/tags when added.

    This test demonstrates that feature_spec_version automatically includes
    any new fields added to TestingFeatureSpec in the future.
    """
    # Create two identical specs
    spec1 = TestingFeatureSpec(
        key=FeatureKey(["test", "metadata"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    spec2 = TestingFeatureSpec(
        key=FeatureKey(["test", "metadata"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    # Their feature_spec_versions should be identical
    assert spec1.feature_spec_version == spec2.feature_spec_version

    # The feature_spec_version uses model_dump(mode="json") which will automatically
    # include any future fields added to the TestingFeatureSpec model
    spec_dict = spec1.model_dump(mode="json")

    # Verify all current fields are included
    assert "key" in spec_dict
    assert "deps" in spec_dict
    assert "fields" in spec_dict

    # When metadata/tags are added in the future, they will automatically
    # be included in spec_dict and thus in feature_spec_version hash


def test_feature_spec_version_with_complex_dependencies() -> None:
    """Test feature_spec_version with complex dependency configurations."""
    graph = FeatureGraph()

    with graph.use():
        # Create upstream features
        class Upstream1(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["upstream", "one"]),
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                ],
            ),
        ):
            pass

        class Upstream2(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["upstream", "two"]),
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                ],
            ),
        ):
            pass

        # Create downstream with complex deps
        class Downstream(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[
                    FeatureDep(
                        feature=FeatureKey(["upstream", "one"]),
                        columns=("col1", "col2"),
                        rename={"col1": "one_col1"},
                    ),
                    FeatureDep(
                        feature=FeatureKey(["upstream", "two"]),
                        columns=None,  # All columns
                        rename=None,
                    ),
                ],
                fields=[
                    FieldSpec(key=FieldKey(["merged"]), code_version=1),
                ],
            ),
        ):
            pass

        # Get feature_spec_version
        feature_spec_version = Downstream.spec().feature_spec_version

        # It should include all the complex dependency configuration
        spec_dict = Downstream.spec().model_dump(mode="json")

        # Verify deps are fully captured
        assert len(spec_dict["deps"]) == 2
        assert spec_dict["deps"][0]["columns"] == ["col1", "col2"]
        assert spec_dict["deps"][0]["rename"] == {"col1": "one_col1"}
        assert spec_dict["deps"][1]["columns"] is None
        assert spec_dict["deps"][1]["rename"] is None

        # The feature_spec_version should be deterministic
        assert feature_spec_version == Downstream.spec().feature_spec_version

        # Create an identical spec directly (not as a Feature class)
        # to verify determinism without registering another feature
        identical_spec = TestingFeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[
                FeatureDep(
                    feature=FeatureKey(["upstream", "one"]),
                    columns=("col1", "col2"),
                    rename={"col1": "one_col1"},
                ),
                FeatureDep(
                    feature=FeatureKey(["upstream", "two"]),
                    columns=None,
                    rename=None,
                ),
            ],
            fields=[
                FieldSpec(key=FieldKey(["merged"]), code_version=1),
            ],
        )

        # Should have same feature_spec_version
        assert identical_spec.feature_spec_version == feature_spec_version
