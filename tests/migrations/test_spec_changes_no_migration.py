"""Test that migration detector uses feature_version, not feature_spec_version.

These tests verify that the migration system correctly:
1. Ignores feature_spec_version changes (for future metadata/tags that don't affect computation)
2. Only triggers on feature_version changes (computational properties)
3. Uses feature_version for all migration detection logic

This is important for:
- Future extensibility (adding tags/metadata to SampleFeatureSpec without triggering migrations)
- Audit trail (feature_spec_version captures complete specification history)
- Migration accuracy (only computational changes require reconciliation)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldKey,
    FieldSpec,
)
from metaxy._testing import TempFeatureModule
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.migrations import detect_diff_migration


def test_feature_spec_version_exists_and_differs_from_feature_version():
    """Verify that feature_spec_version and feature_version are distinct properties.

    This test documents that:
    - feature_spec_version captures ALL SampleFeatureSpec properties
    - feature_version captures only computational properties
    - They can potentially differ (when non-computational properties change)
    """
    # Create a simple feature
    temp_module = TempFeatureModule("test_feature_spec_version")

    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"TestFeature": spec})
    graph = temp_module.graph
    TestFeature = graph.features_by_key[FeatureKey(["test", "feature"])]

    # Both versions should exist
    feature_spec_version = TestFeature.feature_spec_version()
    feature_version = TestFeature.feature_version()

    assert isinstance(feature_spec_version, str)
    assert isinstance(feature_version, str)
    assert len(feature_spec_version) == 64  # SHA256 hex digest
    assert len(feature_version) == 64  # SHA256 hex digest

    # They are currently the same because SampleFeatureSpec has no non-computational properties yet
    # But architecturally they serve different purposes:
    # - feature_spec_version: complete specification hash (for audit trail)
    # - feature_version: computational properties hash (for migration triggering)
    # When SampleFeatureSpec gains metadata/tags, they will differ

    temp_module.cleanup()


def test_migration_detector_uses_feature_version_not_feature_spec_version(
    tmp_path: Path, snapshot: SnapshotAssertion
):
    """Test that migration detection compares feature_version, not feature_spec_version.

    This verifies the core architectural decision:
    - GraphDiffer.diff() compares feature_version fields in snapshots
    - feature_spec_version is stored for audit trail but not used for change detection
    - Only computational changes (affecting feature_version) trigger migrations
    """
    # Create v1: Simple feature with code_version="1"
    temp_v1 = TempFeatureModule("test_migration_detector_v1")

    spec_v1 = SampleFeatureSpec(
        key=FeatureKey(["test", "simple"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_v1.write_features({"Simple": spec_v1})
    graph_v1 = temp_v1.graph
    SimpleV1 = graph_v1.features_by_key[FeatureKey(["test", "simple"])]

    # Setup v1 data and snapshot
    store_path = tmp_path / "delta_store"
    store_v1 = DeltaMetadataStore(root_path=store_path)
    with graph_v1.use(), store_v1:
        data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        store_v1.write_metadata(SimpleV1, data)
        SystemTableStorage(store_v1).push_graph_snapshot()

    # Verify snapshot captures both versions (initialize to satisfy type checker)
    snapshot_data = graph_v1.to_snapshot()
    assert "test/simple" in snapshot_data
    assert "metaxy_feature_version" in snapshot_data["test/simple"]
    assert "metaxy_feature_spec_version" in snapshot_data["test/simple"]

    # Store the versions for comparison
    v1_feature_version: str = snapshot_data["test/simple"]["metaxy_feature_version"]
    v1_feature_spec_version: str = snapshot_data["test/simple"][
        "metaxy_feature_spec_version"
    ]

    # Create v2: Change code_version (affects feature_version)
    temp_v2 = TempFeatureModule("test_migration_detector_v2")

    spec_v2 = SampleFeatureSpec(
        key=FeatureKey(["test", "simple"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="2")
        ],  # Changed! This affects feature_version
    )

    temp_v2.write_features({"Simple": spec_v2})
    graph_v2 = temp_v2.graph
    SimpleV2 = graph_v2.features_by_key[FeatureKey(["test", "simple"])]

    # Verify feature_version changed
    v2_feature_version = SimpleV2.feature_version()
    v2_feature_spec_version = SimpleV2.feature_spec_version()

    assert v1_feature_version != v2_feature_version  # Changed!
    assert (
        v1_feature_spec_version != v2_feature_spec_version
    )  # Also changed (includes code_version)

    # Test migration detection
    # DeltaMetadataStore persists data to disk, so we reuse the same path
    store_v2 = DeltaMetadataStore(root_path=store_path)

    with graph_v2.use(), store_v2:
        # Detect migration (compares latest snapshot vs current graph)
        migration = detect_diff_migration(
            store_v2,
            project="test",  # Changed to match test config
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        # Migration should be detected (feature_version changed)
        assert migration is not None
        assert migration.from_snapshot_version == graph_v1.snapshot_version
        assert migration.to_snapshot_version == graph_v2.snapshot_version

        affected_features = migration.get_affected_features(store_v2, "test")
        assert affected_features == snapshot

    temp_v1.cleanup()
    temp_v2.cleanup()


def test_no_migration_when_only_non_computational_properties_change(tmp_path: Path):
    """Test that changes to non-computational properties don't trigger migrations.

    This test documents the intended behavior for when SampleFeatureSpec gains
    additional properties like metadata, tags, or documentation fields.

    When such properties are added:
    - feature_spec_version will change (captures complete specification)
    - feature_version will NOT change (only computational properties)
    - Migration detector will NOT trigger (compares feature_version)

    Current limitation: SampleFeatureSpec doesn't have such properties yet, so this
    test serves as documentation for future extensibility.
    """
    # Create a feature
    temp_module = TempFeatureModule("test_non_computational")

    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"TestFeature": spec})
    graph = temp_module.graph
    TestFeature = graph.features_by_key[FeatureKey(["test", "feature"])]

    # Setup data and snapshot
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    with graph.use(), store:
        data = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"default": "h1"}],
            }
        )
        store.write_metadata(TestFeature, data)
        SystemTableStorage(store).push_graph_snapshot()

    # Currently, there's no way to change spec without changing feature_version
    # because SampleFeatureSpec only has computational properties
    #
    # In the future, when SampleFeatureSpec has tags/metadata fields:
    # 1. Add tags = ["important", "v2"] to SampleFeatureSpec
    # 2. feature_spec_version would change (hashes ALL properties)
    # 3. feature_version would NOT change (only hashes computational properties)
    # 4. detect_diff_migration() would return None (no feature_version change)
    #
    # This architectural separation ensures:
    # - Rich metadata can be added without forcing data migrations
    # - Complete audit trail via feature_spec_version
    # - Efficient migration detection via feature_version

    # For now, verify that no migration is detected when nothing changes
    with graph.use(), store:
        migration = detect_diff_migration(
            store,
            project="test",  # Changed to match test config
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )
        assert migration is None  # No changes

    temp_module.cleanup()


def test_computational_property_changes_trigger_migrations(
    tmp_path, snapshot: SnapshotAssertion
):
    """Test that all computational property changes trigger migrations.

    Computational properties are those that affect feature_version:
    - Field code_version
    - Dependencies (feature-level and field-level)
    - Field definitions (add/remove fields)

    Changes to these should always trigger migrations.
    """
    test_cases = []

    # Test case 1: code_version change
    temp_cv1 = TempFeatureModule("test_code_version_v1")
    spec_cv1 = SampleFeatureSpec(
        key=FeatureKey(["test", "cv"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )
    temp_cv1.write_features({"Feature": spec_cv1})

    temp_cv2 = TempFeatureModule("test_code_version_v2")
    spec_cv2 = SampleFeatureSpec(
        key=FeatureKey(["test", "cv"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="2")],  # Changed!
    )
    temp_cv2.write_features({"Feature": spec_cv2})

    fv1_cv = temp_cv1.graph.features_by_key[
        FeatureKey(["test", "cv"])
    ].feature_version()
    fv2_cv = temp_cv2.graph.features_by_key[
        FeatureKey(["test", "cv"])
    ].feature_version()

    test_cases.append(
        {
            "name": "code_version_change",
            "changed": fv1_cv != fv2_cv,
        }
    )

    # Test case 2: Adding a field
    temp_f1 = TempFeatureModule("test_field_v1")
    spec_f1 = SampleFeatureSpec(
        key=FeatureKey(["test", "field"]),
        fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
    )
    temp_f1.write_features({"Feature": spec_f1})

    temp_f2 = TempFeatureModule("test_field_v2")
    spec_f2 = SampleFeatureSpec(
        key=FeatureKey(["test", "field"]),
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version="1"),
            FieldSpec(key=FieldKey(["field2"]), code_version="1"),  # Added!
        ],
    )
    temp_f2.write_features({"Feature": spec_f2})

    fv1_f = temp_f1.graph.features_by_key[
        FeatureKey(["test", "field"])
    ].feature_version()
    fv2_f = temp_f2.graph.features_by_key[
        FeatureKey(["test", "field"])
    ].feature_version()

    test_cases.append(
        {
            "name": "field_added",
            "changed": fv1_f != fv2_f,
        }
    )

    # Test case 3: Changing dependencies
    temp_d1 = TempFeatureModule("test_dep_v1")

    upstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "upstream"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    downstream_v1_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        # No deps
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_d1.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_v1_spec}
    )

    temp_d2 = TempFeatureModule("test_dep_v2")

    downstream_v2_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test", "upstream"]))],  # Added dep!
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_d2.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_v2_spec}
    )

    fv1_d = temp_d1.graph.features_by_key[
        FeatureKey(["test", "downstream"])
    ].feature_version()
    fv2_d = temp_d2.graph.features_by_key[
        FeatureKey(["test", "downstream"])
    ].feature_version()

    test_cases.append(
        {
            "name": "dependency_added",
            "changed": fv1_d != fv2_d,
        }
    )

    # Verify all computational changes affected feature_version
    assert test_cases == snapshot

    # Cleanup
    temp_cv1.cleanup()
    temp_cv2.cleanup()
    temp_f1.cleanup()
    temp_f2.cleanup()
    temp_d1.cleanup()
    temp_d2.cleanup()


def test_snapshot_stores_both_versions(tmp_path: Path):
    """Test that graph snapshots store both feature_version and feature_spec_version.

    This ensures:
    - Complete audit trail via feature_spec_version
    - Migration detection via feature_version
    - Future extensibility when they diverge
    """
    temp_module = TempFeatureModule("test_snapshot_versions")

    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"TestFeature": spec})
    graph = temp_module.graph
    TestFeature = graph.features_by_key[FeatureKey(["test", "feature"])]

    # Create store and record snapshot
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    with graph.use(), store:
        data = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"default": "h1"}],
            }
        )
        store.write_metadata(TestFeature, data)
        SystemTableStorage(store).push_graph_snapshot()

        # Check snapshot data structure
        snapshot_data = graph.to_snapshot()
        feature_data = snapshot_data["test/feature"]

        # Both versions should be present
        assert "metaxy_feature_version" in feature_data
        assert "metaxy_feature_spec_version" in feature_data
        assert "feature_spec" in feature_data
        assert "feature_class_path" in feature_data

        # Verify they're valid hashes
        assert len(feature_data["metaxy_feature_version"]) == 64
        assert len(feature_data["metaxy_feature_spec_version"]) == 64

        # Check that they match the class methods
        assert feature_data["metaxy_feature_version"] == TestFeature.feature_version()
        assert (
            feature_data["metaxy_feature_spec_version"]
            == TestFeature.feature_spec_version()
        )

    temp_module.cleanup()


def test_graph_differ_compares_feature_version_not_feature_spec_version():
    """Test that GraphDiffer.diff() uses feature_version for change detection.

    This is the core of the migration system - it should only consider
    feature_version when deciding if a feature has changed.
    """
    from metaxy.graph.diff.differ import GraphDiffer

    differ = GraphDiffer()

    # Create two snapshots with identical feature_version but different feature_spec_version
    # (This will be possible when SampleFeatureSpec has non-computational properties)

    # For now, we verify the diff logic uses the feature_version field
    snapshot1 = {
        "test/feature": {
            "metaxy_feature_version": "abc123",  # Same
            "metaxy_feature_spec_version": "spec_v1",  # Different (if we had metadata/tags)
            "feature_spec": {
                "key": ["test", "feature"],
                "deps": None,
                "fields": [{"key": ["default"], "code_version": 1}],
            },
            "fields": {"default": "field_v1"},
        }
    }

    snapshot2 = {
        "test/feature": {
            "metaxy_feature_version": "abc123",  # Same - no computational change
            "metaxy_feature_spec_version": "spec_v2",  # Different - non-computational change
            "feature_spec": {
                "key": ["test", "feature"],
                "deps": None,
                "fields": [{"key": ["default"], "code_version": 1}],
                # In future: "tags": ["important", "v2"] would change feature_spec_version only
            },
            "fields": {"default": "field_v1"},
        }
    }

    # Compute diff
    diff = differ.diff(snapshot1, snapshot2, "snap1", "snap2")

    # No changes should be detected (feature_version is identical)
    assert len(diff.added_nodes) == 0
    assert len(diff.removed_nodes) == 0
    assert len(diff.changed_nodes) == 0  # Key assertion!
    assert not diff.has_changes

    # Now test with feature_version change
    snapshot3 = {
        "test/feature": {
            "metaxy_feature_version": "xyz789",  # Changed - computational change
            "metaxy_feature_spec_version": "spec_v3",  # Also changed
            "feature_spec": {
                "key": ["test", "feature"],
                "deps": None,
                "fields": [
                    {"key": ["default"], "code_version": 2}
                ],  # code_version changed
            },
            "fields": {"default": "field_v2"},
        }
    }

    diff2 = differ.diff(snapshot1, snapshot3, "snap1", "snap3")

    # Changes should be detected (feature_version differs)
    assert len(diff2.changed_nodes) == 1
    assert diff2.has_changes
    assert diff2.changed_nodes[0].feature_key == FeatureKey(["test", "feature"])
    assert diff2.changed_nodes[0].old_version == "abc123"
    assert diff2.changed_nodes[0].new_version == "xyz789"
