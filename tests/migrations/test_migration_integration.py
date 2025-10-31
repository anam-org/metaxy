"""Integration tests for new migration system with multi-snapshot scenarios.

These tests use TempFeatureModule to create realistic graph evolution scenarios
and test the full migration workflow: detect → execute → verify.
"""

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    InMemoryMetadataStore,
    TestingFeatureSpec,
)
from metaxy._testing import TempFeatureModule
from metaxy._utils import collect_to_polars
from metaxy.config import MetaxyConfig
from metaxy.metadata_store.system_tables import SystemTableStorage
from metaxy.migrations import MigrationExecutor, detect_migration
from metaxy.models.feature import FeatureGraph


@pytest.fixture(autouse=True)
def setup_default_config():
    """Set up default MetaxyConfig for all tests so features use project='default'."""
    config = MetaxyConfig(project="default", stores={})
    MetaxyConfig.set(config)
    yield
    MetaxyConfig.reset()


def migrate_store_to_graph(
    source_store: InMemoryMetadataStore,
    target_graph: FeatureGraph,
) -> InMemoryMetadataStore:
    """Create new store with target graph context but source store's data.

    This includes system tables (snapshots, migrations, events) so that
    migration detection can find the previous snapshot.
    """
    new_store = InMemoryMetadataStore()
    # Copy all storage including system tables
    new_store._storage = source_store._storage.copy()
    # System tables are already copied since they're part of _storage
    return new_store


@pytest.fixture
def simple_graph_v1():
    """Simple graph with one feature."""
    temp_module = TempFeatureModule("test_integration_simple_v1")

    spec_v1 = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "simple"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    temp_module.write_features({"Simple": spec_v1})
    yield temp_module.graph
    temp_module.cleanup()


@pytest.fixture
def simple_graph_v2():
    """Simple graph with code_version changed."""
    temp_module = TempFeatureModule("test_integration_simple_v2")

    spec_v2 = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "simple"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=2),  # Changed!
        ],
    )

    temp_module.write_features({"Simple": spec_v2})
    yield temp_module.graph
    temp_module.cleanup()


@pytest.fixture
def upstream_downstream_v1():
    """Graph with upstream and downstream features."""
    temp_module = TempFeatureModule("test_integration_chain_v1")

    upstream_spec = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "upstream"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=1),
        ],
    )

    downstream_spec = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test_integration", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test_integration", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
    yield temp_module.graph
    temp_module.cleanup()


@pytest.fixture
def upstream_downstream_v2():
    """Graph with upstream code_version changed."""
    temp_module = TempFeatureModule("test_integration_chain_v2")

    upstream_spec = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "upstream"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version=2),  # Changed!
        ],
    )

    downstream_spec = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test_integration", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test_integration", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
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
    store_v1 = InMemoryMetadataStore()
    SimpleV1 = simple_graph_v1.features_by_key[
        FeatureKey(["test_integration", "simple"])
    ]

    with simple_graph_v1.use(), store_v1:
        # Write data
        data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "data_version": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        store_v1.write_metadata(SimpleV1, data)

        # Record v1 snapshot
        store_v1.record_feature_graph_snapshot()

    # Step 2: Migrate to v2 graph
    store_v2 = migrate_store_to_graph(store_v1, simple_graph_v2)
    SimpleV2 = simple_graph_v2.features_by_key[
        FeatureKey(["test_integration", "simple"])
    ]

    with simple_graph_v2.use(), store_v2:
        # Step 3: Detect migration (BEFORE recording v2 snapshot)
        # This compares latest snapshot in store (v1) with active graph (v2)
        migration = detect_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        assert migration is not None
        assert migration.from_snapshot_version == simple_graph_v1.snapshot_version
        assert migration.to_snapshot_version == simple_graph_v2.snapshot_version

        # Snapshot migration structure
        affected_features = migration.get_affected_features(store_v2, "default")
        migration_summary = {
            "description": migration.get_description(
                store_v2, "default"
            ),  # Use get_description() for auto-generation
            "affected_features_count": len(affected_features),
            "affected_features": sorted(affected_features),
        }
        assert migration_summary == snapshot(name="migration_detection")

        # Step 4: Record v2 snapshot (BEFORE executing migration)
        # Migration needs both snapshots to be recorded
        store_v2.record_feature_graph_snapshot()

        # Step 5: Execute migration
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        result = executor.execute(migration, store_v2, project="default", dry_run=False)

        # Root feature should fail (cannot be auto-reconciled)
        assert result.status == "failed"
        assert result.features_completed == 0
        assert result.features_failed == 1
        assert "test_integration/simple" in result.errors
        assert (
            "Root features have user-defined data_versions"
            in result.errors["test_integration/simple"]
        )

        # Snapshot result
        result_summary = {
            "status": result.status,
            "features_completed": result.features_completed,
            "features_failed": result.features_failed,
            "has_root_feature_error": "Root features" in str(result.errors),
        }
        assert result_summary == snapshot(name="migration_result")

        # Step 6: Verify data unchanged (root feature cannot be reconciled)
        final_data = collect_to_polars(
            store_v2.read_metadata(SimpleV2, current_only=False)
        )

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
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = upstream_downstream_v1.features_by_key[
        FeatureKey(["test_integration", "upstream"])
    ]
    DownstreamV1 = upstream_downstream_v1.features_by_key[
        FeatureKey(["test_integration", "downstream"])
    ]

    with upstream_downstream_v1.use(), store_v1:
        # Write upstream (root feature)
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "data_version": [
                    {"default": "h1"},
                    {"default": "h2"},
                    {"default": "h3"},
                ],
            }
        )
        store_v1.write_metadata(UpstreamV1, upstream_data)

        # Write downstream (derived feature)
        downstream_samples = pl.DataFrame({"sample_uid": [1, 2, 3]})
        diff = store_v1.resolve_update(DownstreamV1, sample_df=downstream_samples)
        if len(diff.added) > 0:
            store_v1.write_metadata(DownstreamV1, diff.added)

        # Record v1 snapshot
        store_v1.record_feature_graph_snapshot()

    # Step 2: Migrate to v2 graph
    store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
    UpstreamV2 = upstream_downstream_v2.features_by_key[
        FeatureKey(["test_integration", "upstream"])
    ]
    upstream_downstream_v2.features_by_key[
        FeatureKey(["test_integration", "downstream"])
    ]

    with upstream_downstream_v2.use(), store_v2:
        # Step 3: Detect migration (before recording v2 snapshot)
        migration = detect_migration(
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
        store_v2.record_feature_graph_snapshot()

        # Step 5: Simulate user manually updating upstream (root feature)
        # This is what user must do when root features change
        new_upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "data_version": [
                    {"default": "new_h1"},
                    {"default": "new_h2"},
                    {"default": "new_h3"},
                ],
            }
        )
        store_v2.write_metadata(UpstreamV2, new_upstream_data)

        # Step 6: Execute migration (will reconcile downstream)
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        result = executor.execute(migration, store_v2, project="default", dry_run=False)

        # Upstream will fail (root feature), downstream should succeed
        assert result.status == "failed"  # Because upstream is root
        assert "test_integration/upstream" in result.errors
        assert (
            "Root features have user-defined data_versions"
            in result.errors["test_integration/upstream"]
        )

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
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = upstream_downstream_v1.features_by_key[
        FeatureKey(["test_integration", "upstream"])
    ]
    DownstreamV1 = upstream_downstream_v1.features_by_key[
        FeatureKey(["test_integration", "downstream"])
    ]

    with upstream_downstream_v1.use(), store_v1:
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "data_version": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        store_v1.write_metadata(UpstreamV1, upstream_data)

        downstream_samples = pl.DataFrame({"sample_uid": [1, 2]})
        diff = store_v1.resolve_update(DownstreamV1, sample_df=downstream_samples)
        if len(diff.added) > 0:
            store_v1.write_metadata(DownstreamV1, diff.added)

        store_v1.record_feature_graph_snapshot()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
    UpstreamV2 = upstream_downstream_v2.features_by_key[
        FeatureKey(["test_integration", "upstream"])
    ]
    upstream_downstream_v2.features_by_key[
        FeatureKey(["test_integration", "downstream"])
    ]

    with upstream_downstream_v2.use(), store_v2:
        # Update upstream manually
        new_upstream = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "data_version": [{"default": "new_h1"}, {"default": "new_h2"}],
            }
        )
        store_v2.write_metadata(UpstreamV2, new_upstream)

        # Create downstream-only migration (detect before recording v2 snapshot)
        migration = detect_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )
        assert migration is not None

        # Record v2 snapshot before executing
        store_v2.record_feature_graph_snapshot()

        # Test idempotency using the detected migration (includes both upstream + downstream)
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        # Execute first time - will fail on upstream (root feature) but succeed on downstream
        result1 = executor.execute(
            migration, store_v2, project="default", dry_run=False
        )
        assert result1.status == "failed"  # Upstream will fail
        assert result1.features_completed == 1  # Downstream should complete
        assert result1.features_failed == 1  # Upstream should fail

        # Execute second time (should skip already-completed downstream)
        result2 = executor.execute(
            migration, store_v2, project="default", dry_run=False
        )
        assert result2.status == "failed"  # Still fails on upstream
        assert result2.features_completed == 1  # Downstream was skipped (already done)
        assert result2.features_failed == 1  # Upstream still fails


def test_migration_dry_run(
    tmp_path,
    upstream_downstream_v1: FeatureGraph,
    upstream_downstream_v2: FeatureGraph,
):
    """Test dry-run mode doesn't modify data."""
    # Setup v1 data
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = upstream_downstream_v1.features_by_key[
        FeatureKey(["test_integration", "upstream"])
    ]
    DownstreamV1 = upstream_downstream_v1.features_by_key[
        FeatureKey(["test_integration", "downstream"])
    ]

    with upstream_downstream_v1.use(), store_v1:
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "data_version": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        store_v1.write_metadata(UpstreamV1, upstream_data)

        downstream_samples = pl.DataFrame({"sample_uid": [1, 2]})
        diff = store_v1.resolve_update(DownstreamV1, sample_df=downstream_samples)
        if len(diff.added) > 0:
            store_v1.write_metadata(DownstreamV1, diff.added)

        store_v1.record_feature_graph_snapshot()

        # Get initial data
        initial_data = collect_to_polars(
            store_v1.read_metadata(DownstreamV1, current_only=False)
        )

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
    UpstreamV2 = upstream_downstream_v2.features_by_key[
        FeatureKey(["test_integration", "upstream"])
    ]
    DownstreamV2 = upstream_downstream_v2.features_by_key[
        FeatureKey(["test_integration", "downstream"])
    ]

    with upstream_downstream_v2.use(), store_v2:
        # Update upstream
        new_upstream = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "data_version": [{"default": "new_h1"}, {"default": "new_h2"}],
            }
        )
        store_v2.write_metadata(UpstreamV2, new_upstream)

        # Detect and execute with dry_run=True (detect before recording v2 snapshot)
        migration = detect_migration(
            store_v2,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )
        assert migration is not None

        # Record v2 snapshot before executing
        store_v2.record_feature_graph_snapshot()

        # Test dry-run mode
        storage = SystemTableStorage(store_v2)
        executor = MigrationExecutor(storage)

        result = executor.execute(migration, store_v2, project="default", dry_run=True)

        assert result.status == "skipped"
        assert result.rows_affected > 0  # Would affect rows for downstream, but skipped

        # Verify data unchanged
        final_data = collect_to_polars(
            store_v2.read_metadata(DownstreamV2, current_only=False)
        )

        assert len(final_data) == len(initial_data)
        # Compare data_versions (dict types can't be in sets, so compare directly)
        final_dvs = final_data["data_version"].to_list()
        initial_dvs = initial_data["data_version"].to_list()
        assert final_dvs == initial_dvs


def test_field_dependency_change(tmp_path):
    """Test migration when field-level dependencies change."""
    # Create v1: Downstream depends on both upstream fields
    temp_v1 = TempFeatureModule("test_field_change_v1")

    upstream_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "upstream"]),
        deps=None,
        fields=[
            FieldSpec(key=FieldKey(["frames"]), code_version=1),
            FieldSpec(key=FieldKey(["audio"]), code_version=1),
        ],
    )

    downstream_v1_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream"]),
                        fields=[FieldKey(["frames"]), FieldKey(["audio"])],  # Both
                    )
                ],
            )
        ],
    )

    temp_v1.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_v1_spec}
    )
    graph_v1 = temp_v1.graph

    # Create v2: Downstream only depends on frames
    temp_v2 = TempFeatureModule("test_field_change_v2")

    downstream_v2_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,  # Same code_version
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream"]),
                        fields=[FieldKey(["frames"])],  # Only frames!
                    )
                ],
            )
        ],
    )

    temp_v2.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_v2_spec}
    )
    graph_v2 = temp_v2.graph

    # Setup v1 data
    store_v1 = InMemoryMetadataStore()
    UpstreamV1 = graph_v1.features_by_key[FeatureKey(["test", "upstream"])]
    DownstreamV1 = graph_v1.features_by_key[FeatureKey(["test", "downstream"])]

    with graph_v1.use(), store_v1:
        # Write upstream with both fields
        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1],
                "data_version": [{"frames": "hf", "audio": "ha"}],
            }
        )
        store_v1.write_metadata(UpstreamV1, upstream_data)

        # Write downstream
        downstream_samples = pl.DataFrame({"sample_uid": [1]})
        diff = store_v1.resolve_update(DownstreamV1, sample_df=downstream_samples)
        if len(diff.added) > 0:
            store_v1.write_metadata(DownstreamV1, diff.added)

        store_v1.record_feature_graph_snapshot()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)
    graph_v2.features_by_key[FeatureKey(["test", "upstream"])]
    graph_v2.features_by_key[FeatureKey(["test", "downstream"])]

    with graph_v2.use(), store_v2:
        # Detect migration (before recording v2 snapshot)
        migration = detect_migration(
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

    upstream_a_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "upstream_a"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    upstream_b_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "upstream_b"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    downstream_v1_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream_a"]))],  # Depends on A
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream_a"]),
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

    downstream_v2_spec = TestingFeatureSpec(
        key=FeatureKey(["test", "downstream"]),
        deps=[FeatureDep(key=FeatureKey(["test", "upstream_b"]))],  # Changed to B!
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version=1,
                deps=[
                    FieldDep(
                        feature_key=FeatureKey(["test", "upstream_b"]),
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
    down_v1 = graph_v1.features_by_key[FeatureKey(["test", "downstream"])]
    down_v2 = graph_v2.features_by_key[FeatureKey(["test", "downstream"])]
    assert down_v1.feature_version() != down_v2.feature_version()

    # Setup v1 data
    store_v1 = InMemoryMetadataStore()
    upstream_a_v1 = graph_v1.features_by_key[FeatureKey(["test", "upstream_a"])]
    upstream_b_v1 = graph_v1.features_by_key[FeatureKey(["test", "upstream_b"])]
    graph_v1.features_by_key[FeatureKey(["test", "downstream"])]

    with graph_v1.use(), store_v1:
        # Write both upstreams
        store_v1.write_metadata(
            upstream_a_v1,
            pl.DataFrame({"sample_uid": [1], "data_version": [{"default": "ha"}]}),
        )
        store_v1.write_metadata(
            upstream_b_v1,
            pl.DataFrame({"sample_uid": [1], "data_version": [{"default": "hb"}]}),
        )

        # Write downstream (depends on A in v1)
        downstream_samples = pl.DataFrame({"sample_uid": [1]})
        diff = store_v1.resolve_update(down_v1, sample_df=downstream_samples)
        if len(diff.added) > 0:
            store_v1.write_metadata(down_v1, diff.added)

        store_v1.record_feature_graph_snapshot()

    # Migrate to v2
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)
    graph_v2.features_by_key[FeatureKey(["test", "upstream_a"])]
    graph_v2.features_by_key[FeatureKey(["test", "upstream_b"])]
    graph_v2.features_by_key[FeatureKey(["test", "downstream"])]

    with graph_v2.use(), store_v2:
        # Detect migration (before recording v2 snapshot)
        migration = detect_migration(
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
    store = InMemoryMetadataStore()
    SimpleV1 = simple_graph_v1.features_by_key[
        FeatureKey(["test_integration", "simple"])
    ]

    with simple_graph_v1.use(), store:
        # Write data and record snapshot
        data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "data_version": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        store.write_metadata(SimpleV1, data)
        store.record_feature_graph_snapshot()

        # Try to detect migration (same graph)
        migration = detect_migration(
            store,
            project="default",
            ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            migrations_dir=tmp_path / "migrations",
        )

        assert migration is None  # No changes


def test_migration_with_new_feature(tmp_path, simple_graph_v1: FeatureGraph):
    """Test that adding a new feature doesn't trigger migration for it."""
    # Setup v1 data
    store_v1 = InMemoryMetadataStore()
    SimpleV1 = simple_graph_v1.features_by_key[
        FeatureKey(["test_integration", "simple"])
    ]

    with simple_graph_v1.use(), store_v1:
        data = pl.DataFrame(
            {
                "sample_uid": [1],
                "data_version": [{"default": "h1"}],
            }
        )
        store_v1.write_metadata(SimpleV1, data)
        store_v1.record_feature_graph_snapshot()

    # Create v2 with additional feature
    temp_v2 = TempFeatureModule("test_new_feature_v2")

    simple_spec = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "simple"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],  # Unchanged
    )

    new_spec = TestingFeatureSpec(
        key=FeatureKey(["test_integration", "new"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    temp_v2.write_features({"Simple": simple_spec, "New": new_spec})
    graph_v2 = temp_v2.graph

    # Migrate store
    store_v2 = migrate_store_to_graph(store_v1, graph_v2)

    with graph_v2.use(), store_v2:
        store_v2.record_feature_graph_snapshot()

        # Detect migration
        # Use the project from one of the features in the graph
        migration = detect_migration(
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
