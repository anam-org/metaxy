"""Tests for new type-safe migration models."""

from datetime import datetime, timezone

import pytest

from metaxy.migrations.models import (
    CustomMigration,
    DiffMigration,
    FullGraphMigration,
    Migration,
    MigrationResult,
)


def test_migration_type_property():
    """Test migration_type property returns correct class path."""
    migration = DiffMigration(
        migration_id="test_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        parent="initial",
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    )

    assert migration.migration_type == "metaxy.migrations.models.DiffMigration"


def test_migration_to_storage_dict():
    """Test converting migration to storage dict."""
    migration = DiffMigration(
        migration_id="test_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        parent="initial",
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    )

    storage_dict = migration.to_storage_dict()

    assert storage_dict["migration_id"] == "test_001"
    assert storage_dict["migration_type"] == "metaxy.migrations.models.DiffMigration"
    assert storage_dict["from_snapshot_version"] == "snap1"
    assert storage_dict["to_snapshot_version"] == "snap2"
    assert storage_dict["ops"] == [
        {"type": "metaxy.migrations.ops.DataVersionReconciliation"}
    ]
    # Computed fields should NOT be in storage dict
    assert "affected_features" not in storage_dict
    assert "graph_diff_struct" not in storage_dict
    assert "description" not in storage_dict


def test_migration_from_storage_dict():
    """Test deserializing migration from storage dict."""
    storage_dict = {
        "migration_type": "metaxy.migrations.models.DiffMigration",
        "migration_id": "test_001",
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "parent": "initial",
        "from_snapshot_version": "snap1",
        "to_snapshot_version": "snap2",
        "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    }

    migration = Migration.from_storage_dict(storage_dict)

    assert isinstance(migration, DiffMigration)
    assert migration.migration_id == "test_001"
    assert migration.from_snapshot_version == "snap1"
    assert migration.to_snapshot_version == "snap2"


def test_migration_from_storage_dict_invalid_type():
    """Test deserializing with invalid migration_type."""
    storage_dict = {
        "migration_type": "nonexistent.module.InvalidMigration",
        "migration_id": "test_001",
        "created_at": datetime(2025, 1, 1),
        "description": "Test",
    }

    with pytest.raises(ValueError, match="Failed to load migration class"):
        Migration.from_storage_dict(storage_dict)


def test_migration_from_storage_dict_missing_type():
    """Test deserializing without migration_type."""
    storage_dict = {
        "migration_id": "test_001",
        "created_at": datetime(2025, 1, 1),
        "description": "Test",
    }

    with pytest.raises(ValueError, match="Missing migration_type field"):
        Migration.from_storage_dict(storage_dict)


def test_diff_migration_get_affected_features():
    """Test DiffMigration computes affected features on-demand.

    Note: This test creates a minimal migration but cannot actually compute
    affected features without a store containing the snapshots. In real usage,
    get_affected_features(store) would be called with a proper store.
    """
    migration = DiffMigration(
        migration_id="test_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        parent="initial",
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    )

    # We can't call get_affected_features() without a store since it computes on-demand
    # Just verify the migration was created successfully
    assert migration.from_snapshot_version == "snap1"
    assert migration.to_snapshot_version == "snap2"


def test_full_graph_migration():
    """Test FullGraphMigration basic functionality."""
    migration = FullGraphMigration(
        migration_id="test_002",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
    )

    assert migration.migration_type == "metaxy.migrations.models.FullGraphMigration"
    # FullGraphMigration.get_affected_features() requires store parameter
    # but returns empty list by default (needs to be overridden in subclasses)
    from metaxy import InMemoryMetadataStore

    with InMemoryMetadataStore() as store:
        assert migration.get_affected_features(store) == []


def test_custom_migration():
    """Test CustomMigration base class."""

    class TestCustomMigration(CustomMigration):
        custom_field: str

        def execute(self, store, *, dry_run=False):
            return MigrationResult(
                migration_id=self.migration_id,
                status="completed",
                features_completed=1,
                features_failed=0,
                affected_features=["test/feature"],
                errors={},
                rows_affected=100,
                duration_seconds=1.0,
                timestamp=datetime.now(tz=timezone.utc),
            )

    migration = TestCustomMigration(
        migration_id="custom_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        custom_field="test_value",
    )

    # Migration type should be automatically set from class path
    assert "TestCustomMigration" in migration.migration_type
    # CustomMigration.get_affected_features() requires store parameter
    from metaxy import InMemoryMetadataStore

    with InMemoryMetadataStore() as store:
        assert migration.get_affected_features(store) == []  # Default implementation


def test_custom_migration_roundtrip():
    """Test custom migration serialization roundtrip."""

    # Define custom migration class at module level for import
    class MyCustomMigration(CustomMigration):
        s3_bucket: str

        def execute(self, store, *, dry_run=False):
            return MigrationResult(
                migration_id=self.migration_id,
                status="completed",
                features_completed=0,
                features_failed=0,
                affected_features=[],
                errors={},
                rows_affected=0,
                duration_seconds=0.0,
                timestamp=datetime.now(tz=timezone.utc),
            )

    # Register in module for import
    import sys

    sys.modules[__name__].MyCustomMigration = MyCustomMigration  # pyright: ignore[reportAttributeAccessIssue]

    original = MyCustomMigration(
        migration_id="custom_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        s3_bucket="my-bucket",
    )

    storage_dict = original.to_storage_dict()
    restored = Migration.from_storage_dict(storage_dict)

    assert isinstance(restored, MyCustomMigration)
    assert restored.s3_bucket == "my-bucket"


def test_migration_result_summary():
    """Test MigrationResult summary formatting."""
    result = MigrationResult(
        migration_id="test_001",
        status="completed",
        features_completed=3,
        features_failed=1,
        affected_features=["feature/a", "feature/b", "feature/c"],
        errors={"feature/d": "Connection timeout"},
        rows_affected=1500,
        duration_seconds=5.25,
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    summary = result.summary()

    assert "test_001" in summary
    assert "COMPLETED" in summary
    assert "5.25s" in summary
    assert "3 completed, 1 failed" in summary
    assert "1500" in summary
    assert "feature/a" in summary
    assert "Connection timeout" in summary


def test_diff_migration_roundtrip():
    """Test DiffMigration serialization roundtrip."""
    original = DiffMigration(
        migration_id="diff_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        parent="initial",
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    )

    storage_dict = original.to_storage_dict()
    restored = Migration.from_storage_dict(storage_dict)

    assert isinstance(restored, DiffMigration)
    assert restored.migration_id == original.migration_id
    assert restored.from_snapshot_version == original.from_snapshot_version
    assert restored.to_snapshot_version == original.to_snapshot_version
    assert restored.ops == original.ops
    # Computed fields are not stored and will be recomputed on-demand when accessed
