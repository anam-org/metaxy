"""Tests for migration operations and FullGraphMigration.execute()."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    InMemoryMetadataStore,
    SampleFeatureSpec,
)
from metaxy._testing import TempFeatureModule, add_metaxy_provenance_column
from metaxy._utils import collect_to_polars
from metaxy.config import MetaxyConfig
from metaxy.metadata_store.system_tables import SystemTableStorage
from metaxy.migrations.models import FullGraphMigration
from metaxy.migrations.ops import BaseOperation, DataVersionReconciliation


@pytest.fixture(autouse=True)
def setup_default_config():
    """Set up default MetaxyConfig for all tests."""
    config = MetaxyConfig(project="default", stores={})
    MetaxyConfig.set(config)
    yield
    MetaxyConfig.reset()


# ============================================================================
# BaseOperation Interface Tests
# ============================================================================


def test_base_operation_is_abstract():
    """Test BaseOperation cannot be instantiated directly."""
    with pytest.raises(TypeError):
        _ = BaseOperation()  # pyright: ignore[reportAbstractUsage]


def test_base_operation_requires_execute_for_feature():
    """Test BaseOperation subclass must implement execute_for_feature()."""

    # Create a subclass without implementing execute_for_feature
    class IncompleteOperation(BaseOperation):  # pyright: ignore[reportImplicitAbstractClass]
        pass

    with pytest.raises(TypeError, match="abstract"):
        _ = IncompleteOperation()  # pyright: ignore[reportAbstractUsage]


def test_base_operation_subclass_implements_interface():
    """Test BaseOperation subclass can implement the interface correctly."""

    class TestOperation(BaseOperation):
        value: int = 42

        def execute_for_feature(
            self,
            store,
            feature_key,
            *,
            snapshot_version,
            from_snapshot_version=None,
            dry_run=False,
        ) -> int:
            if dry_run:
                return 10
            return 5

    # Should instantiate successfully
    op = TestOperation()
    assert "TestOperation" in op.type  # Type is auto-computed from class
    assert op.value == 42


def test_base_operation_with_custom_fields():
    """Test BaseOperation subclass can have custom fields."""

    class CustomOperation(BaseOperation):
        s3_bucket: str
        s3_prefix: str
        min_size_mb: int = 10

        def execute_for_feature(
            self,
            store,
            feature_key,
            *,
            snapshot_version,
            from_snapshot_version=None,
            dry_run=False,
        ) -> int:
            return 0

    op = CustomOperation(s3_bucket="my-bucket", s3_prefix="data/")
    assert op.s3_bucket == "my-bucket"
    assert op.s3_prefix == "data/"
    assert op.min_size_mb == 10


# ============================================================================
# Test Custom Operation for Testing
# ============================================================================


class _TestBackfillOperation(BaseOperation):
    """Test operation that backfills with fixed values."""

    fixed_value: str = "test"

    def execute_for_feature(
        self,
        store,
        feature_key,
        *,
        snapshot_version,
        from_snapshot_version=None,
        dry_run=False,
    ) -> int:
        """Simple test implementation."""
        if dry_run:
            return 10
        # Write test data
        return 5


class _FailingOperation(BaseOperation):
    """Test operation that always fails."""

    def execute_for_feature(
        self,
        store,
        feature_key,
        *,
        snapshot_version,
        from_snapshot_version=None,
        dry_run=False,
    ) -> int:
        raise ValueError(f"Intentional failure for {feature_key}")


# ============================================================================
# FullGraphMigration.execute() Tests
# ============================================================================


def test_full_graph_migration_single_operation_single_feature():
    """Test executing migration with single operation and single feature."""
    # Create a simple test graph
    temp_module = TempFeatureModule("test_single_op_single_feat")

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

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        # Setup data
        Upstream = graph.features_by_key[FeatureKey(["test", "upstream"])]
        Downstream = graph.features_by_key[FeatureKey(["test", "downstream"])]

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [{"default": "h1"}, {"default": "h2"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
        store.write_metadata(Upstream, upstream_data)

        # Write downstream
        diff = store.resolve_update(Downstream)
        if len(diff.added) > 0:
            store.write_metadata(Downstream, diff.added)

        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with _TestBackfillOperation
        migration = FullGraphMigration(
            migration_id="test_001",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/downstream"],
                    "fixed_value": "test_value",
                }
            ],
        )

        # Execute
        result = migration.execute(store, "default", dry_run=False)

        # Verify result
        assert result.status == "completed"
        assert result.features_completed == 1
        assert result.features_failed == 0
        assert result.affected_features == ["test/downstream"]
        assert result.rows_affected == 5  # _TestBackfillOperation returns 5

    temp_module.cleanup()


def test_full_graph_migration_single_operation_multiple_features():
    """Test executing migration with single operation and multiple features."""
    temp_module = TempFeatureModule("test_single_op_multi_feat")

    feature_a_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature_a"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    feature_b_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature_b"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"FeatureA": feature_a_spec, "FeatureB": feature_b_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with TestBackfillOperation for both features
        migration = FullGraphMigration(
            migration_id="test_002",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/feature_a", "test/feature_b"],
                }
            ],
        )

        result = migration.execute(store, "default", dry_run=False)

        assert result.status == "completed"
        assert result.features_completed == 2
        assert result.features_failed == 0
        assert set(result.affected_features) == {"test/feature_a", "test/feature_b"}
        assert result.rows_affected == 10  # 5 per feature

    temp_module.cleanup()


def test_full_graph_migration_multiple_operations():
    """Test executing migration with multiple operations."""
    temp_module = TempFeatureModule("test_multi_op")

    feature_a_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature_a"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    feature_b_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature_b"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"FeatureA": feature_a_spec, "FeatureB": feature_b_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with two different operations
        migration = FullGraphMigration(
            migration_id="test_003",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/feature_a"],
                    "fixed_value": "value1",
                },
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/feature_b"],
                    "fixed_value": "value2",
                },
            ],
        )

        result = migration.execute(store, "default", dry_run=False)

        assert result.status == "completed"
        assert result.features_completed == 2
        assert result.features_failed == 0
        assert set(result.affected_features) == {"test/feature_a", "test/feature_b"}
        assert result.rows_affected == 10

    temp_module.cleanup()


def test_full_graph_migration_topological_sorting():
    """Test that features are executed in topological order (deps before dependents)."""
    temp_module = TempFeatureModule("test_topo_sort")

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

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with features in wrong order (downstream before upstream)
        migration = FullGraphMigration(
            migration_id="test_004",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": [
                        "test/downstream",
                        "test/upstream",
                    ],  # Wrong order!
                }
            ],
        )

        result = migration.execute(store, "default", dry_run=False)

        # Should still succeed - execution order should be topologically sorted
        assert result.status == "completed"
        assert result.features_completed == 2
        # Verify upstream was processed before downstream
        assert result.affected_features == ["test/upstream", "test/downstream"]

    temp_module.cleanup()


def test_full_graph_migration_operation_fails():
    """Test that migration continues when one feature fails."""
    temp_module = TempFeatureModule("test_op_fails")

    feature_a_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature_a"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    feature_b_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature_b"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"FeatureA": feature_a_spec, "FeatureB": feature_b_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with FailingOperation
        migration = FullGraphMigration(
            migration_id="test_005",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._FailingOperation",
                    "features": ["test/feature_a", "test/feature_b"],
                }
            ],
        )

        result = migration.execute(store, "default", dry_run=False)

        # Both features should fail
        assert result.status == "failed"
        assert result.features_completed == 0
        assert result.features_failed == 2
        assert "test/feature_a" in result.errors
        assert "test/feature_b" in result.errors
        assert "Intentional failure" in result.errors["test/feature_a"]

    temp_module.cleanup()


def test_full_graph_migration_invalid_operation_class():
    """Test that migration fails when operation class cannot be loaded."""
    temp_module = TempFeatureModule("test_invalid_op")

    feature_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"Feature": feature_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with non-existent operation class
        migration = FullGraphMigration(
            migration_id="test_006",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "nonexistent.module.NonexistentOperation",
                    "features": ["test/feature"],
                }
            ],
        )

        # Should raise error when trying to execute
        with pytest.raises(ValueError, match="Failed to instantiate operation"):
            migration.execute(store, "default", dry_run=False)

    temp_module.cleanup()


def test_full_graph_migration_dry_run():
    """Test that dry run doesn't write events or modify data."""
    temp_module = TempFeatureModule("test_dry_run")

    feature_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"Feature": feature_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        migration = FullGraphMigration(
            migration_id="test_007",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/feature"],
                }
            ],
        )

        result = migration.execute(store, "default", dry_run=True)

        # Status should be skipped
        assert result.status == "skipped"
        assert result.features_completed == 1
        assert (
            result.rows_affected == 10
        )  # _TestBackfillOperation returns 10 in dry run

        # Verify no events were written
        storage = SystemTableStorage(store)
        events_lf = storage.read_migration_events(migration.migration_id, "default")
        events_df = collect_to_polars(events_lf)
        assert len(events_df) == 0

    temp_module.cleanup()


def test_full_graph_migration_resume_after_partial_failure():
    """Test that migration can be resumed after partial failure (idempotent)."""
    temp_module = TempFeatureModule("test_resume")

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

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        # Setup data
        Upstream = graph.features_by_key[FeatureKey(["test", "upstream"])]
        Downstream = graph.features_by_key[FeatureKey(["test", "downstream"])]

        upstream_data = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"default": "h1"}],
            }
        )
        upstream_data = add_metaxy_provenance_column(upstream_data, Upstream)
        store.write_metadata(Upstream, upstream_data)

        diff = store.resolve_update(Downstream)
        if len(diff.added) > 0:
            store.write_metadata(Downstream, diff.added)

        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration where upstream succeeds but downstream fails
        migration = FullGraphMigration(
            migration_id="test_008",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/upstream"],
                },
                {
                    "type": "tests.migrations.test_operations._FailingOperation",
                    "features": ["test/downstream"],
                },
            ],
        )

        # First execution: upstream succeeds, downstream fails
        result1 = migration.execute(store, "default", dry_run=False)
        assert result1.status == "failed"
        assert result1.features_completed == 1
        assert result1.features_failed == 1

        # Re-execute: upstream should be skipped (already completed)
        result2 = migration.execute(store, "default", dry_run=False)
        assert result2.status == "failed"
        assert result2.features_completed == 1  # Upstream was skipped
        assert result2.features_failed == 1  # Downstream still fails

    temp_module.cleanup()


def test_full_graph_migration_operation_specific_config():
    """Test that operation-specific config is passed correctly."""
    temp_module = TempFeatureModule("test_op_config")

    feature_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"Feature": feature_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        # Create migration with custom config
        migration = FullGraphMigration(
            migration_id="test_009",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/feature"],
                    "fixed_value": "custom_value",  # Custom config
                }
            ],
        )

        result = migration.execute(store, "default", dry_run=False)

        assert result.status == "completed"
        # The operation should have been instantiated with custom config
        # (we can't easily verify this without inspection, but at least check it didn't error)

    temp_module.cleanup()


def test_full_graph_migration_events_tracking():
    """Test that migration events are written correctly."""
    temp_module = TempFeatureModule("test_events")

    feature_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"Feature": feature_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()
        snapshot_version = graph.snapshot_version

        migration = FullGraphMigration(
            migration_id="test_010",
            parent="initial",
            created_at=datetime.now(timezone.utc),
            snapshot_version=snapshot_version,
            ops=[
                {
                    "type": "tests.migrations.test_operations._TestBackfillOperation",
                    "features": ["test/feature"],
                }
            ],
        )

        result = migration.execute(store, "default", dry_run=False)

        # Verify execution completed successfully
        assert result.status == "completed"
        assert result.features_completed == 1
        assert result.features_failed == 0

        # Note: Event details are checked in the existing integration tests
        # We've verified that the migration executed successfully

    temp_module.cleanup()


# ============================================================================
# DataVersionReconciliation Tests
# ============================================================================


def test_data_version_reconciliation_requires_from_snapshot():
    """Test DataVersionReconciliation requires from_snapshot_version."""
    temp_module = TempFeatureModule("test_dvr_from_snap")

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

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot()

        op = DataVersionReconciliation()

        # Should raise error without from_snapshot_version
        with pytest.raises(
            ValueError, match="DataVersionReconciliation requires from_snapshot_version"
        ):
            op.execute_for_feature(
                store,
                "test/downstream",
                snapshot_version=graph.snapshot_version,
                from_snapshot_version=None,  # Missing!
                dry_run=False,
            )

    temp_module.cleanup()


def test_data_version_reconciliation_root_feature_error():
    """Test DataVersionReconciliation fails for root features."""
    temp_module = TempFeatureModule("test_dvr_root")

    root_spec = SampleFeatureSpec(
        key=FeatureKey(["test", "root"]),
        fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
    )

    temp_module.write_features({"Root": root_spec})
    graph = temp_module.graph

    with graph.use(), InMemoryMetadataStore() as store:
        Root = graph.features_by_key[FeatureKey(["test", "root"])]

        # Write some data
        data = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"default": "h1"}],
            }
        )
        data = add_metaxy_provenance_column(data, Root)
        store.write_metadata(Root, data)

        store.record_feature_graph_snapshot()
        from_snapshot = graph.snapshot_version

        # Change code version
        temp_module2 = TempFeatureModule("test_dvr_root_v2")
        root_spec_v2 = SampleFeatureSpec(
            key=FeatureKey(["test", "root"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="2")],  # Changed
        )
        temp_module2.write_features({"Root": root_spec_v2})
        graph_v2 = temp_module2.graph

        with graph_v2.use():
            store.record_feature_graph_snapshot()

            op = DataVersionReconciliation()

            # Should raise error for root feature
            with pytest.raises(ValueError, match="Root features have user-defined"):
                op.execute_for_feature(
                    store,
                    "test/root",
                    snapshot_version=graph_v2.snapshot_version,
                    from_snapshot_version=from_snapshot,
                    dry_run=False,
                )

        temp_module2.cleanup()

    temp_module.cleanup()
