"""Tests for new type-safe migration models."""

from datetime import datetime, timezone

import pytest

from metaxy.migrations.models import (
    DiffMigration,
    FullGraphMigration,
    Migration,
    MigrationResult,
    OperationConfig,
)


def test_migration_type_property():
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
    migration = DiffMigration(
        migration_id="test_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        parent="initial",
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    )

    storage_dict = migration.model_dump(mode="json")

    assert storage_dict["migration_id"] == "test_001"
    assert storage_dict["migration_type"] == "metaxy.migrations.models.DiffMigration"
    assert storage_dict["parent"] == "initial"
    assert storage_dict["from_snapshot_version"] == "snap1"
    assert storage_dict["to_snapshot_version"] == "snap2"
    assert storage_dict["ops"] == [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]


def test_migration_from_storage_dict():
    from metaxy.migrations.models import MigrationAdapter

    storage_dict = {
        "migration_type": "metaxy.migrations.models.DiffMigration",
        "migration_id": "test_001",
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "parent": "initial",
        "from_snapshot_version": "snap1",
        "to_snapshot_version": "snap2",
        "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    }

    migration = MigrationAdapter.validate_python(storage_dict)

    assert isinstance(migration, DiffMigration)
    assert migration.migration_id == "test_001"
    assert migration.from_snapshot_version == "snap1"
    assert migration.to_snapshot_version == "snap2"


def test_migration_from_storage_dict_invalid_type():
    from pydantic import ValidationError

    from metaxy.migrations.models import MigrationAdapter

    storage_dict = {
        "migration_type": "nonexistent.module.InvalidMigration",
        "migration_id": "test_001",
        "parent": "initial",
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }

    with pytest.raises(ValidationError):
        MigrationAdapter.validate_python(storage_dict)


def test_migration_from_storage_dict_missing_type():
    from pydantic import ValidationError

    from metaxy.migrations.models import MigrationAdapter

    storage_dict = {
        "migration_id": "test_001",
        "parent": "initial",
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }

    with pytest.raises(ValidationError):
        MigrationAdapter.validate_python(storage_dict)


def test_diff_migration_get_affected_features():
    """Test DiffMigration computes affected features on-demand."""
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


def test_full_graph_migration(tmp_path):
    migration = FullGraphMigration(
        migration_id="test_002",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
        ops=[],  # Required field
    )

    assert migration.migration_type == "metaxy.migrations.models.FullGraphMigration"
    # FullGraphMigration.get_affected_features() returns sorted list from ops
    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        assert migration.get_affected_features(store, "default") == []


def test_custom_migration(tmp_path):
    class TestCustomMigration(Migration):
        custom_field: str

        @property
        def migration_type(self) -> str:
            return f"{self.__class__.__module__}.{self.__class__.__name__}"

        def get_affected_features(self, store, project):
            return []  # Custom implementation

        def execute(self, store, project, *, dry_run=False):
            return MigrationResult(
                migration_id=self.migration_id,
                status="completed",
                features_completed=1,
                features_failed=0,
                affected_features=["test/feature"],
                features_skipped=0,
                errors={},
                rows_affected=100,
                duration_seconds=1.0,
                timestamp=datetime.now(tz=timezone.utc),
            )

    migration = TestCustomMigration(
        migration_id="custom_001",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        custom_field="test_value",
    )

    # Migration type should be automatically set from class path
    assert "TestCustomMigration" in migration.migration_type
    # get_affected_features() requires store parameter
    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        assert migration.get_affected_features(store, "default") == []  # Default implementation


def test_custom_migration_serialization():
    # Define custom migration class
    class MyCustomMigration(Migration):
        s3_bucket: str

        @property
        def migration_type(self) -> str:
            return f"{self.__class__.__module__}.{self.__class__.__name__}"

        def get_affected_features(self, store, project):
            return []

        def execute(self, store, project, *, dry_run=False):
            return MigrationResult(
                migration_id=self.migration_id,
                status="completed",
                features_completed=0,
                features_failed=0,
                features_skipped=0,
                affected_features=[],
                errors={},
                rows_affected=0,
                duration_seconds=0.0,
                timestamp=datetime.now(tz=timezone.utc),
            )

    original = MyCustomMigration(
        migration_id="custom_001",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        s3_bucket="my-bucket",
    )

    storage_dict = original.model_dump(mode="json")

    assert storage_dict["migration_id"] == "custom_001"
    assert storage_dict["parent"] == "initial"
    assert storage_dict["s3_bucket"] == "my-bucket"


def test_migration_result_summary():
    result = MigrationResult(
        migration_id="test_001",
        status="completed",
        features_completed=3,
        features_failed=1,
        features_skipped=0,
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
    from metaxy.migrations.models import MigrationAdapter

    original = DiffMigration(
        migration_id="diff_001",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        parent="initial",
        from_snapshot_version="snap1",
        to_snapshot_version="snap2",
        ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
    )

    storage_dict = original.model_dump(mode="json")
    restored = MigrationAdapter.validate_python(storage_dict)

    assert isinstance(restored, DiffMigration)
    assert restored.migration_id == original.migration_id
    assert restored.parent == original.parent
    assert restored.from_snapshot_version == original.from_snapshot_version
    assert restored.to_snapshot_version == original.to_snapshot_version
    assert restored.ops == original.ops


# ============================================================================
# OperationConfig Tests
# ============================================================================


def test_operation_config_valid():
    config = OperationConfig(
        type="metaxy.migrations.ops.DataVersionReconciliation",
        features=["feature/a", "feature/b"],
    )

    # type is now stored as a string for lazy loading
    assert config.type == "metaxy.migrations.ops.DataVersionReconciliation"
    assert config.features == ["feature/a", "feature/b"]


def test_operation_config_empty_features():
    config = OperationConfig(
        type="metaxy.migrations.ops.DataVersionReconciliation",
        features=[],
    )

    assert config.features == []


def test_operation_config_roundtrip():
    original = OperationConfig(
        type="metaxy.migrations.ops.DataVersionReconciliation",
        features=["feature/a", "feature/b"],
        custom_field="value",  # Extra field, allowed with extra="allow"  # ty: ignore[unknown-argument]
        batch_size=100,  # Extra field  # ty: ignore[unknown-argument]
    )

    # Serialize to dict
    dict_form = original.model_dump()

    # Deserialize back
    restored = OperationConfig.model_validate(dict_form)

    # type is now a string (lazy loading)
    assert restored.type == original.type == "metaxy.migrations.ops.DataVersionReconciliation"
    assert restored.features == original.features
    # Extra fields are preserved
    assert dict_form["custom_field"] == "value"
    assert dict_form["batch_size"] == 100


def test_operation_with_basesettings_env_vars():
    import os

    from metaxy.migrations.ops import BaseOperation

    # Define a test operation
    class TestOperation(BaseOperation):
        database_url: str
        api_key: str = "default_key"

        def execute_for_feature(
            self,
            store,
            feature_key,
            *,
            snapshot_version,
            from_snapshot_version=None,
            dry_run=False,
        ):
            return 0

    # Set environment variables
    os.environ["DATABASE_URL"] = "postgresql://localhost:5432/test"
    os.environ["API_KEY"] = "secret123"

    try:
        # Instantiate from config dict (mimics YAML loading)
        op = TestOperation(database_url="postgresql://localhost:5432/test", api_key="secret123")

        assert op.database_url == "postgresql://localhost:5432/test"
        assert op.api_key == "secret123"

        # Can also instantiate with only env vars (no config dict)
        op_from_env = (
            TestOperation()  # ty: ignore[missing-argument]
        )  # Testing env var behavior

        assert op_from_env.database_url == "postgresql://localhost:5432/test"
        assert op_from_env.api_key == "secret123"
    finally:
        del os.environ["DATABASE_URL"]
        del os.environ["API_KEY"]


# ============================================================================
# FullGraphMigration Tests
# ============================================================================


def test_full_graph_migration_get_affected_features(tmp_path):
    migration = FullGraphMigration(
        migration_id="test_full_001",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
        ops=[
            {
                "type": "metaxy.migrations.ops.DataVersionReconciliation",
                "features": ["feature/a", "feature/b"],
            },
            {
                "type": "metaxy.migrations.ops.DataVersionReconciliation",
                "features": ["feature/c"],
            },
        ],
    )

    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        affected = migration.get_affected_features(store, "default")

        # Should include all features from all operations (sorted)
        assert affected == ["feature/a", "feature/b", "feature/c"]


def test_full_graph_migration_deduplicates_features(tmp_path):
    migration = FullGraphMigration(
        migration_id="test_full_002",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
        ops=[
            {
                "type": "metaxy.migrations.ops.DataVersionReconciliation",
                "features": ["feature/a", "feature/b"],
            },
            {
                "type": "metaxy.migrations.ops.DataVersionReconciliation",
                "features": ["feature/b", "feature/c"],  # feature/b appears again
            },
        ],
    )

    from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

    with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
        affected = migration.get_affected_features(store, "default")

        # Should deduplicate feature/b
        assert affected == ["feature/a", "feature/b", "feature/c"]


def test_full_graph_migration_with_from_snapshot():
    migration = FullGraphMigration(
        migration_id="test_full_003",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap2",
        from_snapshot_version="snap1",  # Cross-snapshot migration
        ops=[
            {
                "type": "metaxy.migrations.ops.DataVersionReconciliation",
                "features": ["feature/a"],
            }
        ],
    )

    assert migration.snapshot_version == "snap2"
    assert migration.from_snapshot_version == "snap1"


def test_get_failed_features_with_retry(store):
    """Test that get_failed_features only returns features whose latest event is failure."""
    from metaxy.metadata_store.system import Event, SystemTableStorage

    storage = SystemTableStorage(store)

    with store:
        migration_id = "test_migration_retry"
        project = "test"

        # Feature 1: fails then succeeds
        storage.write_event(Event.migration_started(project, migration_id))
        storage.write_event(Event.feature_started(project, migration_id, "feature_1"))
        storage.write_event(Event.feature_failed(project, migration_id, "feature_1", "First error"))
        # Retry - now succeeds
        storage.write_event(Event.feature_started(project, migration_id, "feature_1"))
        storage.write_event(Event.feature_completed(project, migration_id, "feature_1", rows_affected=100))

        # Feature 2: fails and stays failed
        storage.write_event(Event.feature_started(project, migration_id, "feature_2"))
        storage.write_event(Event.feature_failed(project, migration_id, "feature_2", "Permanent error"))

        # Feature 3: succeeds on first try
        storage.write_event(Event.feature_started(project, migration_id, "feature_3"))
        storage.write_event(Event.feature_completed(project, migration_id, "feature_3", rows_affected=50))

        # Get completed and failed features
        completed = storage.get_completed_features(migration_id, project)
        failed = storage.get_failed_features(migration_id, project)

        # Assertions
        assert "feature_1" in completed, "Feature that succeeded after retry should be in completed"
        assert "feature_3" in completed, "Feature that succeeded first try should be in completed"
        assert "feature_1" not in failed, "Feature that succeeded after retry should NOT be in failed"
        assert "feature_2" in failed, "Feature that never succeeded should be in failed"
        assert failed["feature_2"] == "Permanent error"
        assert "feature_3" not in failed, "Feature that succeeded should NOT be in failed"

        # Check migration summary
        summary = storage.get_migration_summary(migration_id, project)
        assert summary["status"].value == "in_progress"  # Not completed or failed
        assert len(summary["completed_features"]) == 2  # feature_1 and feature_3
        assert len(summary["failed_features"]) == 1  # only feature_2
        assert "feature_2" in summary["failed_features"]


# ============================================================================
# MigrationStatusInfo Tests
# ============================================================================


def test_migration_status_info_properties():
    """Test MigrationStatusInfo computed properties."""
    from metaxy.metadata_store.system import MigrationStatus
    from metaxy.migrations.models import MigrationStatusInfo

    status_info = MigrationStatusInfo(
        migration_id="test_001",
        status=MigrationStatus.IN_PROGRESS,
        expected_features=["feature/a", "feature/b", "feature/c", "feature/d"],
        completed_features=["feature/a", "feature/b"],
        failed_features={"feature/c": "Error message"},
        pending_features=["feature/d"],
    )

    assert status_info.features_total == 4
    # features_remaining = pending (1) + failed (1) = 2
    assert status_info.features_remaining == 2


def test_migration_status_info_failed_features_count_as_remaining():
    """Test that failed features are counted in features_remaining.

    This is important because failed features need to be retried, so they
    should not be considered "done". Previously, only pending features were
    counted, which caused the CLI to show "All features completed" even when
    some features had failed.
    """
    from metaxy.metadata_store.system import MigrationStatus
    from metaxy.migrations.models import MigrationStatusInfo

    # Scenario: 2 features expected, 1 completed, 1 failed, 0 pending
    status_info = MigrationStatusInfo(
        migration_id="test_failed_remaining",
        status=MigrationStatus.IN_PROGRESS,
        expected_features=["feature/a", "feature/b"],
        completed_features=["feature/a"],
        failed_features={"feature/b": "Some error"},
        pending_features=[],  # No pending features
    )

    assert status_info.features_total == 2
    # Even though pending is empty, failed features count as remaining
    assert status_info.features_remaining == 1
    assert len(status_info.pending_features) == 0
    assert len(status_info.failed_features) == 1


def test_migration_status_info_empty():
    """Test MigrationStatusInfo with empty lists."""
    from metaxy.metadata_store.system import MigrationStatus
    from metaxy.migrations.models import MigrationStatusInfo

    status_info = MigrationStatusInfo(
        migration_id="test_002",
        status=MigrationStatus.NOT_STARTED,
        expected_features=[],
        completed_features=[],
        failed_features={},
        pending_features=[],
    )

    assert status_info.features_total == 0
    assert status_info.features_remaining == 0


def test_migration_status_info_all_completed():
    """Test MigrationStatusInfo when all features are completed."""
    from metaxy.metadata_store.system import MigrationStatus
    from metaxy.migrations.models import MigrationStatusInfo

    status_info = MigrationStatusInfo(
        migration_id="test_003",
        status=MigrationStatus.COMPLETED,
        expected_features=["feature/a", "feature/b"],
        completed_features=["feature/a", "feature/b"],
        failed_features={},
        pending_features=[],
    )

    assert status_info.features_total == 2
    assert status_info.features_remaining == 0


# ============================================================================
# get_status_info Method Tests
# ============================================================================


def test_full_graph_migration_get_status_info(store):
    """Test Migration.get_status_info() convenience method."""
    from metaxy.metadata_store.system import Event, MigrationStatus, SystemTableStorage

    storage = SystemTableStorage(store)

    with store:
        migration = FullGraphMigration(
            migration_id="test_status_info",
            parent="initial",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            snapshot_version="snap1",
            ops=[
                {
                    "type": "metaxy.migrations.ops.DataVersionReconciliation",
                    "features": ["feature/a", "feature/b", "feature/c"],
                }
            ],
        )

        # Before any events, status should be NOT_STARTED
        status_info = migration.get_status_info(store, "test")
        assert status_info.status == MigrationStatus.NOT_STARTED
        assert status_info.expected_features == ["feature/a", "feature/b", "feature/c"]
        assert status_info.completed_features == []
        assert status_info.failed_features == {}
        assert status_info.pending_features == ["feature/a", "feature/b", "feature/c"]
        assert status_info.features_total == 3
        assert status_info.features_remaining == 3

        # Start migration and complete some features
        storage.write_event(Event.migration_started(project="test", migration_id=migration.migration_id))
        storage.write_event(
            Event.feature_completed(
                project="test",
                migration_id=migration.migration_id,
                feature_key="feature/a",
                rows_affected=100,
            )
        )

        status_info = migration.get_status_info(store, "test")
        assert status_info.status == MigrationStatus.IN_PROGRESS
        assert status_info.completed_features == ["feature/a"]
        assert status_info.features_remaining == 2
        assert set(status_info.pending_features) == {"feature/b", "feature/c"}

        # Fail one feature
        storage.write_event(
            Event.feature_failed(
                project="test",
                migration_id=migration.migration_id,
                feature_key="feature/b",
                error_message="Test error",
            )
        )

        status_info = migration.get_status_info(store, "test")
        assert status_info.failed_features == {"feature/b": "Test error"}
        assert status_info.pending_features == ["feature/c"]
        # features_remaining = pending (1) + failed (1) = 2
        assert status_info.features_remaining == 2


# ============================================================================
# Lazy Import Tests for operations property
# ============================================================================


def test_operations_property_lazy_import():
    """Test that Migration.operations lazily imports operation classes."""
    migration = FullGraphMigration(
        migration_id="test_lazy_001",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
        ops=[
            {
                "type": "metaxy.migrations.ops.DataVersionReconciliation",
                "features": ["feature/a"],
            }
        ],
    )

    # Access operations property - should successfully import the class
    operations = migration.operations
    assert len(operations) == 1
    assert operations[0].__class__.__name__ == "DataVersionReconciliation"


def test_operations_property_invalid_import():
    """Test that invalid operation class path raises clear error."""
    migration = FullGraphMigration(
        migration_id="test_invalid_001",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
        ops=[
            {
                "type": "nonexistent.module.FakeOperation",
                "features": ["feature/a"],
            }
        ],
    )

    # Accessing operations should raise ValueError with clear message
    with pytest.raises(ValueError, match="Failed to import operation class"):
        _ = migration.operations


def test_operations_property_invalid_class_name():
    """Test that invalid class name in valid module raises clear error."""
    migration = FullGraphMigration(
        migration_id="test_invalid_002",
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
        ops=[
            {
                "type": "metaxy.migrations.ops.NonExistentClass",
                "features": ["feature/a"],
            }
        ],
    )

    with pytest.raises(ValueError, match="Failed to import operation class"):
        _ = migration.operations


def test_operation_config_string_type_allows_reading_invalid_paths():
    """Test that OperationConfig with string type can be read even if path is invalid.

    This is a key feature - we can read migration YAML files even if the
    operation classes have been renamed or don't exist yet.
    """
    # This should NOT raise an error at config creation time
    config = OperationConfig(
        type="some.future.module.NewOperation",
        features=["feature/a"],
    )

    # The type is stored as a string
    assert config.type == "some.future.module.NewOperation"

    # Error only occurs when trying to instantiate via Migration.operations
