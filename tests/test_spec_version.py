"""Tests for TestingFeatureSpec.feature_spec_version property."""

import hashlib
import json

from syrupy.assertion import SnapshotAssertion

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldKey,
    FieldSpec,
    TestingFeatureSpec,
)


def test_feature_spec_version_deterministic(snapshot: SnapshotAssertion) -> None:
    """Test that feature_spec_version is deterministic."""
    spec = TestingFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    version1 = spec.feature_spec_version
    version2 = spec.feature_spec_version

    # Should be deterministic
    assert version1 == version2

    # Should be 64 characters (SHA256 hex digest)
    assert len(version1) == 64

    # Should be hex string
    assert all(c in "0123456789abcdef" for c in version1)

    # Snapshot the hash
    assert version1 == snapshot


def test_feature_spec_version_includes_all_properties(
    snapshot: SnapshotAssertion,
) -> None:
    """Test that feature_spec_version includes all specification properties."""
    # Create a complex spec with all properties
    spec = TestingFeatureSpec(
        key=FeatureKey(["complex", "feature"]),
        deps=[
            FeatureDep(
                key=FeatureKey(["upstream", "one"]),
                columns=("col1", "col2"),
                rename={"col1": "renamed_col1"},
            ),
            FeatureDep(
                key=FeatureKey(["upstream", "two"]),
                columns=None,  # All columns
                rename=None,
            ),
        ],
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version=1),
            FieldSpec(key=FieldKey(["field2"]), code_version=2),
        ],
    )

    version = spec.feature_spec_version

    # Verify it's a valid SHA256 hash
    assert len(version) == 64
    assert all(c in "0123456789abcdef" for c in version)

    # Snapshot for stability
    assert version == snapshot


def test_feature_spec_version_changes_with_any_property() -> None:
    """Test that feature_spec_version changes when any property changes."""
    base_spec = TestingFeatureSpec(
        key=FeatureKey(["base", "feature"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )
    base_version = base_spec.feature_spec_version

    # Change key
    spec_key_changed = TestingFeatureSpec(
        key=FeatureKey(["changed", "feature"]),  # Changed!
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )
    assert spec_key_changed.feature_spec_version != base_version

    # Add deps
    spec_deps_added = TestingFeatureSpec(
        key=FeatureKey(["base", "feature"]),
        deps=[FeatureDep(key=FeatureKey(["upstream"]))],  # Added!
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )
    assert spec_deps_added.feature_spec_version != base_version

    # Change field code_version
    spec_field_changed = TestingFeatureSpec(
        key=FeatureKey(["base", "feature"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=2),  # Changed!
        ],
    )
    assert spec_field_changed.feature_spec_version != base_version

    # Add field
    spec_field_added = TestingFeatureSpec(
        key=FeatureKey(["base", "feature"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
            FieldSpec(key=FieldKey(["new_field"]), code_version=1),  # Added!
        ],
    )
    assert spec_field_added.feature_spec_version != base_version


def test_feature_spec_version_consistent_ordering() -> None:
    """Test that feature_spec_version is consistent regardless of field order."""
    # Create specs with fields in different order
    spec1 = TestingFeatureSpec(
        key=FeatureKey(["test", "ordering"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["field_a"]), code_version=1),
            FieldSpec(key=FieldKey(["field_b"]), code_version=2),
        ],
    )

    spec2 = TestingFeatureSpec(
        key=FeatureKey(["test", "ordering"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["field_b"]), code_version=2),  # Reordered
            FieldSpec(key=FieldKey(["field_a"]), code_version=1),
        ],
    )

    # The order of fields in the list matters for the spec
    # So these should have different hashes
    assert spec1.feature_spec_version != spec2.feature_spec_version

    # But creating the same spec multiple times should be deterministic
    spec3 = TestingFeatureSpec(
        key=FeatureKey(["test", "ordering"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["field_a"]), code_version=1),
            FieldSpec(key=FieldKey(["field_b"]), code_version=2),
        ],
    )
    assert spec1.feature_spec_version == spec3.feature_spec_version


def test_feature_spec_version_manual_verification() -> None:
    """Manually verify the feature_spec_version computation."""
    spec = TestingFeatureSpec(
        key=FeatureKey(["manual", "test"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    # Get the spec version
    feature_spec_version = spec.feature_spec_version

    # Manually compute what it should be
    spec_dict = spec.model_dump(mode="json")
    spec_json = json.dumps(spec_dict, sort_keys=True)
    expected_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()

    # They should match
    assert feature_spec_version == expected_hash


def test_feature_spec_version_with_column_selection_and_rename(
    snapshot: SnapshotAssertion,
) -> None:
    """Test feature_spec_version with column selection and renaming in deps."""
    spec = TestingFeatureSpec(
        key=FeatureKey(["test", "columns"]),
        deps=[
            FeatureDep(
                key=FeatureKey(["upstream"]),
                columns=("col1", "col2"),
                rename={"col1": "new_col1", "col2": "new_col2"},
            )
        ],
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    version = spec.feature_spec_version

    # Should be valid SHA256
    assert len(version) == 64
    assert all(c in "0123456789abcdef" for c in version)

    # Snapshot for stability
    assert version == snapshot

    # Changing column selection should change the hash
    spec_different_columns = TestingFeatureSpec(
        key=FeatureKey(["test", "columns"]),
        deps=[
            FeatureDep(
                key=FeatureKey(["upstream"]),
                columns=("col1", "col3"),  # Different columns!
                rename={"col1": "new_col1", "col3": "new_col3"},
            )
        ],
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )
    assert spec_different_columns.feature_spec_version != version

    # Changing rename mapping should change the hash
    spec_different_rename = TestingFeatureSpec(
        key=FeatureKey(["test", "columns"]),
        deps=[
            FeatureDep(
                key=FeatureKey(["upstream"]),
                columns=("col1", "col2"),
                rename={"col1": "different_name"},  # Different rename!
            )
        ],
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )
    assert spec_different_rename.feature_spec_version != version


def test_feature_feature_spec_version_classmethod() -> None:
    """Test that Feature.feature_spec_version() classmethod works correctly."""
    from metaxy import Feature, FeatureGraph

    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["test", "classmethod"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        # Get feature_spec_version via classmethod
        classmethod_version = TestFeature.feature_spec_version()

        # Get feature_spec_version directly from spec
        direct_version = TestFeature.spec().feature_spec_version

        # They should be identical
        assert classmethod_version == direct_version

        # Should be valid SHA256
        assert len(classmethod_version) == 64
        assert all(c in "0123456789abcdef" for c in classmethod_version)


def test_feature_spec_version_stored_in_snapshot(snapshot: SnapshotAssertion) -> None:
    """Test that feature_spec_version is included in graph snapshots."""
    from metaxy import Feature, FeatureGraph

    graph = FeatureGraph()

    with graph.use():

        class SnapshotFeature(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["snapshot", "test"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                ],
            ),
        ):
            pass

        # Get snapshot dict
        snapshot_dict = graph.to_snapshot()

        # Should have the feature
        feature_key_str = "snapshot/test"
        assert feature_key_str in snapshot_dict

        # Should contain feature_spec_version
        feature_data = snapshot_dict[feature_key_str]
        assert "feature_spec_version" in feature_data

        # feature_spec_version should match the Feature's feature_spec_version
        assert (
            feature_data["feature_spec_version"]
            == SnapshotFeature.spec().feature_spec_version
        )

        # feature_spec_version should be different from feature_version
        assert feature_data["feature_spec_version"] != feature_data["feature_version"]

        # Snapshot the entire feature data for stability
        assert feature_data == snapshot


def test_feature_spec_version_recorded_in_metadata_store(
    snapshot: SnapshotAssertion,
) -> None:
    """Test that feature_spec_version is recorded when pushing to metadata store."""

    from metaxy import Feature, FeatureGraph
    from metaxy.metadata_store import InMemoryMetadataStore

    graph = FeatureGraph()

    with graph.use():

        class RecordedFeature(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["recorded", "feature"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        store = InMemoryMetadataStore()

        with store:
            # Record the feature graph snapshot
            result = store.record_feature_graph_snapshot()

            is_existing = result.already_recorded

            # Should be a new snapshot
            assert not is_existing

            # Read the feature_versions system table
            features_df = store.read_features(current=True)

            # Should have one feature
            assert len(features_df) == 1

            # Check that feature_spec_version column exists and has value
            assert "feature_spec_version" in features_df.columns

            # Get row as dict
            feature_row = features_df.to_dicts()[0]
            assert feature_row["feature_spec_version"] is not None
            assert len(feature_row["feature_spec_version"]) == 64

            # feature_spec_version should match the Feature's feature_spec_version
            assert (
                feature_row["feature_spec_version"]
                == RecordedFeature.spec().feature_spec_version
            )

            # feature_spec_version should be different from feature_version
            assert feature_row["feature_spec_version"] != feature_row["feature_version"]

            # Snapshot for stability
            assert {
                "feature_key": feature_row["feature_key"],
                "feature_version": feature_row["feature_version"],
                "feature_spec_version": feature_row["feature_spec_version"],
                "snapshot_version": feature_row["snapshot_version"],
            } == snapshot


def test_feature_spec_version_idempotent_snapshot_recording() -> None:
    """Test that recording the same snapshot twice preserves feature_spec_version."""
    from metaxy import Feature, FeatureGraph
    from metaxy.metadata_store import InMemoryMetadataStore

    graph = FeatureGraph()

    with graph.use():

        class IdempotentFeature(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["idempotent", "test"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        store = InMemoryMetadataStore()

        with store:
            # First push
            result = store.record_feature_graph_snapshot()

            snapshot_v1 = result.snapshot_version

            is_existing_1 = result.already_recorded
            assert not is_existing_1

            features_df_1 = store.read_features(current=True)
            feature_spec_version_1 = features_df_1.to_dicts()[0]["feature_spec_version"]

            # Second push (identical graph)
            result = store.record_feature_graph_snapshot()

            snapshot_v2 = result.snapshot_version

            is_existing_2 = result.already_recorded
            assert is_existing_2  # Should detect existing snapshot
            assert snapshot_v1 == snapshot_v2  # Same snapshot version

            features_df_2 = store.read_features(current=True)
            feature_spec_version_2 = features_df_2.to_dicts()[0]["feature_spec_version"]

            # feature_spec_version should be identical
            assert feature_spec_version_1 == feature_spec_version_2


def test_feature_spec_version_different_from_feature_version_always() -> None:
    """Test that feature_spec_version is always different from feature_version.

    These two hashes serve different purposes and use different computation methods:
    - feature_spec_version: Direct JSON serialization of TestingFeatureSpec (all properties)
    - feature_version: Graph-based computation including dependency chains
    """
    from metaxy import Feature, FeatureGraph

    graph = FeatureGraph()

    with graph.use():
        # Test with root feature (no deps)
        class RootFeature(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["root"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        # Test with downstream feature (with deps)
        class DownstreamFeature(
            Feature,
            spec=TestingFeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(key=FeatureKey(["root"]))],
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        # Both should have different feature_spec_version vs feature_version
        assert RootFeature.feature_spec_version() != RootFeature.feature_version()
        assert (
            DownstreamFeature.feature_spec_version()
            != DownstreamFeature.feature_version()
        )

        # Both versions should be valid SHA256 hashes
        for feature_cls in [RootFeature, DownstreamFeature]:
            spec_v = feature_cls.feature_spec_version()
            feature_v = feature_cls.feature_version()

            assert len(spec_v) == 64
            assert len(feature_v) == 64
            assert all(c in "0123456789abcdef" for c in spec_v)
            assert all(c in "0123456789abcdef" for c in feature_v)


def test_feature_spec_version_with_multiple_complex_deps(
    snapshot: SnapshotAssertion,
) -> None:
    """Test feature_spec_version with multiple complex dependencies."""
    spec = TestingFeatureSpec(
        key=FeatureKey(["complex", "multi", "dep"]),
        deps=[
            FeatureDep(
                key=FeatureKey(["upstream", "one"]),
                columns=("a", "b", "c"),
                rename={"a": "upstream1_a", "b": "upstream1_b"},
            ),
            FeatureDep(
                key=FeatureKey(["upstream", "two"]),
                columns=(),  # Empty tuple = only system columns
                rename=None,
            ),
            FeatureDep(
                key=FeatureKey(["upstream", "three"]),
                columns=None,  # None = all columns
                rename={"col": "renamed_col"},
            ),
        ],
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version=1),
            FieldSpec(key=FieldKey(["field2"]), code_version=2),
            FieldSpec(key=FieldKey(["field3"]), code_version=3),
        ],
    )

    version = spec.feature_spec_version

    # Should be valid SHA256
    assert len(version) == 64
    assert all(c in "0123456789abcdef" for c in version)

    # Snapshot the hash
    assert version == snapshot

    # Verify the spec is deterministic
    spec_copy = TestingFeatureSpec(
        key=FeatureKey(["complex", "multi", "dep"]),
        deps=[
            FeatureDep(
                key=FeatureKey(["upstream", "one"]),
                columns=("a", "b", "c"),
                rename={"a": "upstream1_a", "b": "upstream1_b"},
            ),
            FeatureDep(
                key=FeatureKey(["upstream", "two"]),
                columns=(),
                rename=None,
            ),
            FeatureDep(
                key=FeatureKey(["upstream", "three"]),
                columns=None,
                rename={"col": "renamed_col"},
            ),
        ],
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version=1),
            FieldSpec(key=FieldKey(["field2"]), code_version=2),
            FieldSpec(key=FieldKey(["field3"]), code_version=3),
        ],
    )

    assert spec_copy.feature_spec_version == version
