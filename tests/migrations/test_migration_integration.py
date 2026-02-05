"""Integration tests for new migration system with multi-snapshot scenarios.

These tests use TempFeatureModule to create realistic graph evolution scenarios
and test the full migration workflow: detect → execute → verify.
"""

from datetime import datetime, timezone

import polars as pl
import pytest
from metaxy_testing import TempFeatureModule, add_metaxy_provenance_column
from metaxy_testing.models import SampleFeatureSpec
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
)
from metaxy._utils import collect_to_polars
from metaxy.config import MetaxyConfig
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.migrations import MigrationExecutor, detect_diff_migration
from metaxy.models.feature import FeatureGraph


@pytest.fixture(autouse=True)
def setup_default_config():
    """Set up default MetaxyConfig for all tests so features use project='default'."""
    config = MetaxyConfig(project="default", stores={})
    MetaxyConfig.set(config)
    yield
    MetaxyConfig.reset()


def migrate_store_to_graph(
    source_store: DeltaMetadataStore,
    target_graph: FeatureGraph,
) -> DeltaMetadataStore:
    """Create new store with target graph context but source store's data.

    This includes system tables (snapshots, migrations, events) so that
    migration detection can find the previous snapshot.

    For DeltaMetadataStore, we just create a new store pointing to the same
    root_path since data is persisted on disk.
    """
    # DeltaMetadataStore persists data to disk, so we can just create a new
    # instance pointing to the same location
    return DeltaMetadataStore(root_path=source_store._root_uri)


@pytest.fixture
def simple_graph_v1():
    """Simple graph with one feature."""
    temp_module = TempFeatureModule("test_integration_simple_v1")

    spec_v1 = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "simple"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="1"),
        ],
    )

    temp_module.write_features({"Simple": spec_v1})
    yield temp_module.graph
    temp_module.cleanup()


@pytest.fixture
def simple_graph_v2():
    """Simple graph with code_version changed."""
    temp_module = TempFeatureModule("test_integration_simple_v2")

    spec_v2 = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "simple"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="2"),  # Changed!
        ],
    )

    temp_module.write_features({"Simple": spec_v2})
    yield temp_module.graph
    temp_module.cleanup()


@pytest.fixture
def upstream_downstream_v1():
    """Graph with upstream and downstream features."""
    temp_module = TempFeatureModule("test_integration_chain_v1")

    upstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "upstream"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="1"),
        ],
    )

    downstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test_integration", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test_integration", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    temp_module.write_features({"Upstream": upstream_spec, "Downstream": downstream_spec})
    yield temp_module.graph
    temp_module.cleanup()


@pytest.fixture
def upstream_downstream_v2():
    """Graph with upstream code_version changed."""
    temp_module = TempFeatureModule("test_integration_chain_v2")

    upstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "upstream"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="2"),  # Changed!
        ],
    )

    downstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test_integration", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test_integration", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    temp_module.write_features({"Upstream": upstream_spec, "Downstream": downstream_spec})
    yield temp_module.graph
    temp_module.cleanup()


def test_basic_migration_flow(
    simple_graph_v1: FeatureGraph,
    simple_graph_v2: FeatureGraph,
    snapshot: SnapshotAssertion,
    tmp_path,
):
    """Test basic end-to-end migration flow: detect → execute → verify."""
    # Step 1: Setup v1 data
    store_v1 = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    SimpleV1 = simple_graph_v1.feature_definitions_by_key[FeatureKey(["test_integration", "simple"])]

    with simple_graph_v1.use(), store_v1:
        # Write data
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
        data = add_metaxy_provenance_column(data, SimpleV1)
        store_v1.write(SimpleV1, data)

        # Record v1 snapshot
        SystemTableStorage(store_v1).push_graph_snapshot()

    # Step 2: Migrate to v2 graph
    store_v2 = migrate_store_to_graph(store_v1, simple_graph_v2)
    SimpleV2 = simple_graph_v2.feature_definitions_by_key[FeatureKey(["test_integration", "simple"])]

    with simple_graph_v2.use(), store_v2.open("w"):
        # Step 3: Detect migration (BEFORE recording v2 snapshot)
        # This compares latest snapshot in store (v1) with active graph (v2)
        migration = detect_diff_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        assert migration is not None
        assert migration.from_snapshot_version == simple_graph_v1.get_project_snapshot_version("default")
        assert migration.to_snapshot_version == simple_graph_v2.get_project_snapshot_version("default")

        # Snapshot migration structure
        affected_features = migration.get_affected_features(store_v2, "default")
        migration_summary = {
            "affected_features_count": len(affected_features),
            "affected_features": sorted(affected_features),
        }
        assert migration_summary == snapshot(name="migration_detection")

        # Step 4: Record v2 snapshot (BEFORE executing migration)
        # Migration needs both snapshots to be recorded
        SystemTableStorage(store_v2).push_graph_snapshot()

        # Step 5: Execute migration
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        result = executor.execute(migration, store_v2, project="default", dry_run=False)

        # Root feature should fail (cannot be auto-reconciled)
        assert result.status == "failed"
        assert result.features_completed == 0
        assert result.features_failed == 1
        assert "test_integration/simple" in result.errors
        assert "Root features have user-defined field_provenance" in result.errors["test_integration/simple"]

        # Snapshot result
        result_summary = {
            "status": result.status,
            "features_completed": result.features_completed,
            "features_failed": result.features_failed,
            "has_root_feature_error": "Root features" in str(result.errors),
        }
        assert result_summary == snapshot(name="migration_result")

        # Step 6: Verify data unchanged (root feature cannot be reconciled)
        final_data = collect_to_polars(store_v2.read(SimpleV2, with_feature_history=True))

        # Should still have v1 data (root features can't auto-reconcile)
        assert len(final_data) == 3
        assert set(final_data["sample_uid"].to_list()) == {1, 2, 3}


def test_upstream_downstream_migration(
    tmp_path,
    upstream_downstream_v1: FeatureGraph,
    upstream_downstream_v2: FeatureGraph,
    snapshot: SnapshotAssertion,
):
    """Test migration with upstream/downstream dependency chain."""
    # Step 1: Setup v1 data
    store_v1 = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    UpstreamV1 = upstream_downstream_v1.feature_definitions_by_key[FeatureKey(["test_integration", "upstream"])]
    DownstreamV1 = upstream_downstream_v1.feature_definitions_by_key[FeatureKey(["test_integration", "downstream"])]

    with upstream_downstream_v1.use(), store_v1:
        # Write upstream (root feature)
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        # Add metaxy_provenance column using the helper
        upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamV1)
        store_v1.write(UpstreamV1, upstream_data)

        # Write downstream (derived feature)
        # Don't provide samples - let system auto-load upstream and calculate provenance_by_field
        diff = store_v1.resolve_update(DownstreamV1)
        if len(diff.added) > 0:
            store_v1.write(DownstreamV1, diff.added)

        # Record v1 snapshot
        SystemTableStorage(store_v1).push_graph_snapshot()

    # Step 2: Migrate to v2 graph
    store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
    UpstreamV2 = upstream_downstream_v2.feature_definitions_by_key[FeatureKey(["test_integration", "upstream"])]
    upstream_downstream_v2.feature_definitions_by_key[FeatureKey(["test_integration", "downstream"])]

    with upstream_downstream_v2.use(), store_v2.open("w"):
        # Step 3: Detect migration (before recording v2 snapshot)
        migration = detect_diff_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        assert migration is not None

        # Both features should be affected (upstream changed → downstream affected)
        affected_features = migration.get_affected_features(store_v2, "default")
        migration_summary = {
            "affected_features_count": len(affected_features),
            "affected_features": sorted(affected_features),
        }
        assert migration_summary == snapshot(name="affected_features")

        # Step 4: Record v2 snapshot (BEFORE executing migration)
        SystemTableStorage(store_v2).push_graph_snapshot()

        # Step 5: Simulate user manually updating upstream (root feature)
        # This is what user must do when root features change
        new_upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "new_h1"},
                    {"default": "new_h2"},
                    {"default": "new_h3"},
                ],
            }
        )
        new_upstream_data = add_metaxy_provenance_column(new_upstream_data, UpstreamV2)
        store_v2.write(UpstreamV2, new_upstream_data)

        # Step 6: Execute migration (will reconcile downstream)
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        result = executor.execute(migration, store_v2, project="default", dry_run=False)

        # Upstream will fail (root feature), downstream should succeed
        assert result.status == "failed"  # Because upstream is root
        assert "test_integration/upstream" in result.errors
        assert "Root features have user-defined field_provenance" in result.errors["test_integration/upstream"]

        # Note: With the new minimal migration API, we can't manually select features to reconcile
        # The migration will try to reconcile ALL affected features (both upstream and downstream)
        # Since upstream is a root feature, it will fail, resulting in overall "failed" status
        # but downstream should still be reconciled successfully

        # In a real scenario, users would handle root features manually and only
        # execute migrations for derived features


def test_migration_idempotency(
    tmp_path,
    upstream_downstream_v1: FeatureGraph,
    upstream_downstream_v2: FeatureGraph,
):
    """Test that migrations can be re-run safely (idempotent)."""
    # Setup v1 data with downstream
    store_v1 = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    UpstreamV1 = upstream_downstream_v1.feature_definitions_by_key[FeatureKey(["test_integration", "upstream"])]
    DownstreamV1 = upstream_downstream_v1.feature_definitions_by_key[FeatureKey(["test_integration", "downstream"])]

    with upstream_downstream_v1.use(), store_v1:
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamV1)
        store_v1.write(UpstreamV1, upstream_data)

        # Write downstream - let system auto-load upstream and calculate provenance_by_field
        diff = store_v1.resolve_update(DownstreamV1)
        if len(diff.added) > 0:
            store_v1.write(DownstreamV1, diff.added)

        SystemTableStorage(store_v1).push_graph_snapshot()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
    UpstreamV2 = upstream_downstream_v2.feature_definitions_by_key[FeatureKey(["test_integration", "upstream"])]
    upstream_downstream_v2.feature_definitions_by_key[FeatureKey(["test_integration", "downstream"])]

    with upstream_downstream_v2.use(), store_v2.open("w"):
        # Update upstream manually
        new_upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"default": "new_h1"},
                    {"default": "new_h2"},
                ],
            }
        )
        new_upstream_data = add_metaxy_provenance_column(new_upstream_data, UpstreamV2)
        store_v2.write(UpstreamV2, new_upstream_data)

        # Create downstream-only migration (detect before recording v2 snapshot)
        migration = detect_diff_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )
        assert migration is not None

        # Record v2 snapshot before executing
        SystemTableStorage(store_v2).push_graph_snapshot()

        # Test idempotency using the detected migration (includes both upstream + downstream)
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        # Execute first time - will fail on upstream (root feature) and skip downstream due to dependency
        result1 = executor.execute(migration, store_v2, project="default", dry_run=False)
        assert result1.status == "failed"  # Upstream will fail
        assert result1.features_completed == 0  # Downstream skipped due to failed upstream
        assert result1.features_failed == 1  # Only upstream failed
        assert result1.features_skipped == 1  # Downstream skipped due to failed dependency
        assert "test_integration/upstream" in result1.errors
        assert "test_integration/downstream" in result1.errors
        assert "Skipped due to failed dependencies" in result1.errors["test_integration/downstream"]

        # Execute second time - same result since upstream still fails
        result2 = executor.execute(migration, store_v2, project="default", dry_run=False)
        assert result2.status == "failed"  # Still fails on upstream
        assert result2.features_completed == 0  # Downstream still skipped
        assert result2.features_failed == 1  # Only upstream failed
        assert result2.features_skipped == 1  # Downstream still skipped


def test_migration_dry_run(
    tmp_path,
    upstream_downstream_v1: FeatureGraph,
    upstream_downstream_v2: FeatureGraph,
):
    """Test dry-run mode doesn't modify data."""
    # Setup v1 data
    store_v1 = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    UpstreamV1 = upstream_downstream_v1.feature_definitions_by_key[FeatureKey(["test_integration", "upstream"])]
    DownstreamV1 = upstream_downstream_v1.feature_definitions_by_key[FeatureKey(["test_integration", "downstream"])]

    with upstream_downstream_v1.use(), store_v1:
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamV1)
        store_v1.write(UpstreamV1, upstream_data)

        # Write downstream - let system auto-load upstream and calculate provenance_by_field
        diff = store_v1.resolve_update(DownstreamV1)
        if len(diff.added) > 0:
            store_v1.write(DownstreamV1, diff.added)

        SystemTableStorage(store_v1).push_graph_snapshot()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
    UpstreamV2 = upstream_downstream_v2.feature_definitions_by_key[FeatureKey(["test_integration", "upstream"])]
    DownstreamV2 = upstream_downstream_v2.feature_definitions_by_key[FeatureKey(["test_integration", "downstream"])]

    with upstream_downstream_v2.use(), store_v2.open("w"):
        # Update upstream
        new_upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"default": "new_h1"},
                    {"default": "new_h2"},
                ],
            }
        )
        new_upstream_data = add_metaxy_provenance_column(new_upstream_data, UpstreamV2)
        store_v2.write(UpstreamV2, new_upstream_data)

        # Detect and execute with dry_run=True (detect before recording v2 snapshot)
        migration = detect_diff_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )
        assert migration is not None

        # Record v2 snapshot before executing
        SystemTableStorage(store_v2).push_graph_snapshot()

        # Get initial downstream data AFTER upstream write and snapshot, BEFORE migration
        initial_data = collect_to_polars(store_v2.read(DownstreamV2, with_feature_history=True))

        # Test dry-run mode
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        result = executor.execute(migration, store_v2, project="default", dry_run=True)

        assert result.status == "skipped"
        # In dry-run mode with failing root feature, no rows would be affected
        # (upstream fails, downstream skipped due to dependency)
        assert result.rows_affected == 0
        assert result.features_failed == 1  # Upstream fails even in dry-run
        assert result.features_skipped == 1  # Downstream skipped

        # Verify data unchanged - read in same context
        final_data = collect_to_polars(store_v2.read(DownstreamV2, with_feature_history=True))

        assert len(final_data) == len(initial_data)

        # Sort both DataFrames by sample_uid for deterministic comparison
        initial_sorted = initial_data.sort("sample_uid")
        final_sorted = final_data.sort("sample_uid")

        # Compare sample_uids
        assert initial_sorted["sample_uid"].to_list() == final_sorted["sample_uid"].to_list()

        # Compare field_provenance (now sorted, so order-independent)
        initial_dvs = initial_sorted["metaxy_provenance_by_field"].to_list()
        final_dvs = final_sorted["metaxy_provenance_by_field"].to_list()
        assert final_dvs == initial_dvs


def test_field_dependency_change(tmp_path):
    """Test migration when field-level dependencies change."""
    # Create v1: Downstream depends on both upstream fields
    temp_v1 = TempFeatureModule("test_field_change_v1")

    upstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "upstream"]),
        fields=[
            FieldSpec(key=FieldKey(["frames"]), code_version="1"),
            FieldSpec(key=FieldKey(["audio"]), code_version="1"),
        ],
    )

    downstream_v1_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test", "upstream"]),
                        fields=[FieldKey(["frames"]), FieldKey(["audio"])],  # Both
                    )
                ],
            )
        ],
    )

    temp_v1.write_features({"Upstream": upstream_spec, "Downstream": downstream_v1_spec})
    graph_v1 = temp_v1.graph

    # Create v2: Downstream only depends on frames
    temp_v2 = TempFeatureModule("test_field_change_v2")

    downstream_v2_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",  # Same code_version
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test", "upstream"]),
                        fields=[FieldKey(["frames"])],  # Only frames!
                    )
                ],
            )
        ],
    )

    temp_v2.write_features({"Upstream": upstream_spec, "Downstream": downstream_v2_spec})
    graph_v2 = temp_v2.graph

    # Setup v1 data
    store_v1 = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    UpstreamV1 = graph_v1.feature_definitions_by_key[FeatureKey(["test", "upstream"])]
    DownstreamV1 = graph_v1.feature_definitions_by_key[FeatureKey(["test", "downstream"])]

    with graph_v1.use(), store_v1:
        # Write upstream with both fields
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "hf", "audio": "ha"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, UpstreamV1)
        store_v1.write(UpstreamV1, upstream_data)

        # Write downstream
        # Write downstream - let system auto-load upstream and calculate provenance_by_field
        diff = store_v1.resolve_update(DownstreamV1)
        if len(diff.added) > 0:
            store_v1.write(DownstreamV1, diff.added)

        SystemTableStorage(store_v1).push_graph_snapshot()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)
    graph_v2.feature_definitions_by_key[FeatureKey(["test", "upstream"])]
    graph_v2.feature_definitions_by_key[FeatureKey(["test", "downstream"])]

    with graph_v2.use(), store_v2:
        # Detect migration (before recording v2 snapshot)
        migration = detect_diff_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        assert migration is not None
        # Downstream should be detected (field deps changed)
        affected_features = migration.get_affected_features(store_v2, "default")
        assert "test/downstream" in affected_features

    temp_v1.cleanup()
    temp_v2.cleanup()


def test_feature_dependency_swap(tmp_path):
    """Test migration when feature swaps which upstream it depends on."""
    # Create v1: Downstream depends on UpstreamA
    temp_v1 = TempFeatureModule("test_dep_swap_v1")

    upstream_a_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "upstream_a"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    upstream_b_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "upstream_b"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    downstream_v1_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test", "upstream_a"]))],  # Depends on A
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test", "upstream_a"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    temp_v1.write_features(
        {
            "UpstreamA": upstream_a_spec,
            "UpstreamB": upstream_b_spec,
            "Downstream": downstream_v1_spec,
        }
    )
    graph_v1 = temp_v1.graph

    # Create v2: Downstream now depends on UpstreamB
    temp_v2 = TempFeatureModule("test_dep_swap_v2")

    downstream_v2_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test", "upstream_b"]))],  # Changed to B!
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test", "upstream_b"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    temp_v2.write_features(
        {
            "UpstreamA": upstream_a_spec,
            "UpstreamB": upstream_b_spec,
            "Downstream": downstream_v2_spec,
        }
    )
    graph_v2 = temp_v2.graph

    # Verify feature_versions differ
    downstream_key = FeatureKey(["test", "downstream"])
    assert graph_v1.get_feature_version(downstream_key) != graph_v2.get_feature_version(downstream_key)

    # Setup v1 data
    store_v1 = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    upstream_a_key = FeatureKey(["test", "upstream_a"])
    upstream_b_key = FeatureKey(["test", "upstream_b"])

    with graph_v1.use(), store_v1:
        # Write both upstreams
        data_a = pl.DataFrame({"sample_uid": [1], "metaxy_provenance_by_field": [{"default": "ha"}]})
        data_a = add_metaxy_provenance_column(data_a, upstream_a_key)
        store_v1.write(upstream_a_key, data_a)

        data_b = pl.DataFrame({"sample_uid": [1], "metaxy_provenance_by_field": [{"default": "hb"}]})
        data_b = add_metaxy_provenance_column(data_b, upstream_b_key)
        store_v1.write(upstream_b_key, data_b)

        # Write downstream (depends on A in v1)
        # Let system auto-load upstream and calculate provenance_by_field
        diff = store_v1.resolve_update(downstream_key)
        if len(diff.added) > 0:
            store_v1.write(downstream_key, diff.added)

        SystemTableStorage(store_v1).push_graph_snapshot()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)

    with graph_v2.use(), store_v2:
        # Detect migration (before recording v2 snapshot)
        migration = detect_diff_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        assert migration is not None
        # Downstream should be affected (dependency changed)
        affected_features = migration.get_affected_features(store_v2, "default")
        assert "test/downstream" in affected_features

    temp_v1.cleanup()
    temp_v2.cleanup()


def test_no_changes_detected(tmp_path, simple_graph_v1: FeatureGraph):
    """Test that no migration is generated when nothing changed."""
    store = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    simple_key = FeatureKey(["test_integration", "simple"])

    with simple_graph_v1.use(), store:
        # Write data and record snapshot
        data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        data = add_metaxy_provenance_column(data, simple_key)
        store.write(simple_key, data)
        SystemTableStorage(store).push_graph_snapshot()

        # Try to detect migration (same graph)
        migration = detect_diff_migration(
            store,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        assert migration is None  # No changes


def test_migration_with_new_feature(tmp_path, simple_graph_v1: FeatureGraph):
    """Test that adding a new feature doesn't trigger migration for it."""
    # Setup v1 data
    store_v1 = DeltaMetadataStore(root_path=tmp_path / "delta_store")
    SimpleV1 = simple_graph_v1.feature_definitions_by_key[FeatureKey(["test_integration", "simple"])]

    with simple_graph_v1.use(), store_v1:
        data = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"default": "h1"}],
            }
        )
        data = add_metaxy_provenance_column(data, SimpleV1)
        store_v1.write(SimpleV1, data)
        SystemTableStorage(store_v1).push_graph_snapshot()

    # Create v2 with additional feature
    temp_v2 = TempFeatureModule("test_new_feature_v2")

    simple_spec = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "simple"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],  # Unchanged
    )

    new_spec = SampleFeatureSpec(
        key=FeatureKey(["test_integration", "new"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_v2.write_features({"Simple": simple_spec, "New": new_spec})
    graph_v2 = temp_v2.graph

    # Migrate store
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)

    with graph_v2.use(), store_v2:
        SystemTableStorage(store_v2).push_graph_snapshot()

        # Detect migration
        # Use the project from one of the features in the graph
        migration = detect_diff_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        # Should be None or not include the new feature
        # (new features have no existing data to migrate)
        if migration is not None:
            affected_features = migration.get_affected_features(store_v2, "default")
            assert "test_integration/new" not in affected_features

    temp_v2.cleanup()


# ============================================================================
# Multi-Operation Migration Integration Tests
# ============================================================================


def test_full_graph_migration_integration(tmp_path):
    """Test end-to-end FullGraphMigration with real feature graph and store."""
    from metaxy.migrations.models import FullGraphMigration

    # Create test graph
    temp_module = TempFeatureModule("test_full_graph_migration")

    upstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "upstream"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    downstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            )
        ],
    )

    temp_module.write_features({"Upstream": upstream_spec, "Downstream": downstream_spec})
    graph = temp_module.graph

    with graph.use(), DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        # Setup initial data
        Upstream = graph.feature_definitions_by_key[FeatureKey(["test", "upstream"])]
        Downstream = graph.feature_definitions_by_key[FeatureKey(["test", "downstream"])]

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
        store.write(Upstream, upstream_data)

        # Write downstream
        diff = store.resolve_update(Downstream)
        if len(diff.added) > 0:
            store.write(Downstream, diff.added)

        SystemTableStorage(store).push_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create FullGraphMigration using the test operation from test_operations
        migration = FullGraphMigration(
            migration_id="integration_001",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/upstream", "test/downstream"],
                    "fixed_value": "test_value",
                }
            ],
        )

        # Execute migration
        result = migration.execute(store, "default", dry_run=False)

        # Verify results
        assert result.status == "completed"
        assert result.features_completed == 2
        assert result.features_failed == 0
        assert set(result.affected_features) == {"test/upstream", "test/downstream"}
        assert result.rows_affected == 10  # 5 per feature

    temp_module.cleanup()


# Note: Additional integration tests for multiple operations, partial failures,
# and cross-snapshot operations are covered in tests/migrations/test_operations.py
# to avoid redundant test operation class definitions.


def test_migration_rerun_flag(tmp_path):
    """Test that rerun=True forces re-execution of completed features."""
    from metaxy.migrations.models import FullGraphMigration

    # Create test graph
    temp_module = TempFeatureModule("test_migration_rerun")

    feature_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "rerun_feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"RerunFeature": feature_spec})
    graph = temp_module.graph

    with graph.use(), DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        # Setup initial data
        Feature = graph.feature_definitions_by_key[FeatureKey(["test", "rerun_feature"])]

        feature_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        feature_data = add_metaxy_provenance_column(feature_data, Feature)
        store.write(Feature, feature_data)

        SystemTableStorage(store).push_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with test operation
        migration = FullGraphMigration(
            migration_id="rerun_test_001",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/rerun_feature"],
                    "fixed_value": "test_value",
                }
            ],
        )

        storage = SystemTableStorage(store)
        executor = MigrationExecutor(storage)

        # Execute first time - should complete successfully
        result1 = executor.execute(migration, store, "default", dry_run=False)
        assert result1.status == "completed"
        assert result1.features_completed == 1
        assert result1.rows_affected == 5  # From test operation

        # Execute second time without rerun - should skip completed feature
        result2 = executor.execute(migration, store, "default", dry_run=False)
        assert result2.status == "completed"
        assert result2.features_completed == 1  # Still counted as completed
        assert result2.rows_affected == 0  # But no new work done (skipped)

        # Execute third time with rerun=True - should re-process
        result3 = executor.execute(migration, store, "default", dry_run=False, rerun=True)
        assert result3.status == "completed"
        assert result3.features_completed == 1
        assert result3.rows_affected == 5  # Work done again

    temp_module.cleanup()


def test_full_graph_migration_empty_operations(tmp_path):
    """Test FullGraphMigration with no operations."""
    temp_module = TempFeatureModule("test_empty_ops")

    feature_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"Feature": feature_spec})
    graph = temp_module.graph

    with graph.use(), DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        SystemTableStorage(store).push_graph_snapshot()
        snapshot_version = graph.snapshot_version

        from metaxy.migrations.models import FullGraphMigration

        migration = FullGraphMigration(
            migration_id="integration_005",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[],  # No operations
        )

        result = migration.execute(store, "default", dry_run=False)

        # Should complete successfully with no work done
        assert result.status == "completed"
        assert result.features_completed == 0
        assert result.features_failed == 0
        assert result.rows_affected == 0

    temp_module.cleanup()
