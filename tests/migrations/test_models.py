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

    storage_dict = migration.model_dump(mode="json")

    assert storage_dict["migration_id"] == "test_001"
    assert storage_dict["migration_type"] == "metaxy.migrations.models.DiffMigration"
    assert storage_dict["parent"] == "initial"
    assert storage_dict["from_snapshot_version"] == "snap1"
    assert storage_dict["to_snapshot_version"] == "snap2"
    assert storage_dict["ops"] == [
        {"type": "metaxy.migrations.ops.DataVersionReconciliation"}
    ]


def test_migration_from_storage_dict():
    """Test deserializing migration from storage dict."""
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
    """Test deserializing with invalid migration_type."""
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
    """Test deserializing without migration_type."""
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
        parent="initial",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        snapshot_version="snap1",
        ops=[],  # Required field
    )

    assert migration.migration_type == "metaxy.migrations.models.FullGraphMigration"
    # FullGraphMigration.get_affected_features() returns sorted list from ops
    from metaxy import InMemoryMetadataStore

    with InMemoryMetadataStore() as store:
        assert migration.get_affected_features(store, "default") == []


def test_custom_migration():
    """Test custom migration by subclassing Migration directly."""

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
    from metaxy import InMemoryMetadataStore

    with InMemoryMetadataStore() as store:
        assert (
            migration.get_affected_features(store, "default") == []
        )  # Default implementation


def test_custom_migration_serialization():
    """Test custom migration can be serialized with model_dump."""

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
    """Test MigrationResult summary formatting."""
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
    """Test DiffMigration serialization roundtrip."""
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
    """Test creating a valid OperationConfig."""
    from metaxy.migrations.ops import DataVersionReconciliation

    config = OperationConfig(
        type="metaxy.migrations.ops.DataVersionReconciliation",
        features=["feature/a", "feature/b"],
    )

    assert config.type == DataVersionReconciliation
    assert config.features == ["feature/a", "feature/b"]


def test_operation_config_empty_features():
    """Test OperationConfig allows empty features list."""
    config = OperationConfig(
        type="metaxy.migrations.ops.DataVersionReconciliation",
        features=[],
    )

    assert config.features == []


def test_operation_config_roundtrip():
    """Test OperationConfig serialization and deserialization with extra fields."""
    from metaxy.migrations.ops import DataVersionReconciliation

    original = OperationConfig(
        type="metaxy.migrations.ops.DataVersionReconciliation",
        features=["feature/a", "feature/b"],
        custom_field="value",  # pyright: ignore[reportCallIssue]  # Extra field, allowed with extra="allow"
        batch_size=100,  # pyright: ignore[reportCallIssue]  # Extra field
    )

    # Serialize to dict
    dict_form = original.model_dump()

    # Deserialize back
    restored = OperationConfig.model_validate(dict_form)

    assert restored.type == original.type == DataVersionReconciliation
    assert restored.features == original.features
    # Extra fields are preserved
    assert dict_form["custom_field"] == "value"
    assert dict_form["batch_size"] == 100


def test_operation_with_basesettings_env_vars():
    """Test operation classes can read from environment variables using BaseSettings."""
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
        op = TestOperation(
            database_url="postgresql://localhost:5432/test", api_key="secret123"
        )

        assert op.database_url == "postgresql://localhost:5432/test"
        assert op.api_key == "secret123"

        # Can also instantiate with only env vars (no config dict)
        op_from_env = TestOperation()  # pyright: ignore[reportCallIssue]  # Testing env var behavior

        assert op_from_env.database_url == "postgresql://localhost:5432/test"
        assert op_from_env.api_key == "secret123"
    finally:
        del os.environ["DATABASE_URL"]
        del os.environ["API_KEY"]


# ============================================================================
# FullGraphMigration Tests
# ============================================================================


def test_full_graph_migration_get_affected_features():
    """Test FullGraphMigration.get_affected_features() aggregates from all operations."""
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

    from metaxy import InMemoryMetadataStore

    with InMemoryMetadataStore() as store:
        affected = migration.get_affected_features(store, "default")

        # Should include all features from all operations (sorted)
        assert affected == ["feature/a", "feature/b", "feature/c"]


def test_full_graph_migration_deduplicates_features():
    """Test FullGraphMigration deduplicates features across operations."""
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

    from metaxy import InMemoryMetadataStore

    with InMemoryMetadataStore() as store:
        affected = migration.get_affected_features(store, "default")

        # Should deduplicate feature/b
        assert affected == ["feature/a", "feature/b", "feature/c"]


def test_full_graph_migration_with_from_snapshot():
    """Test FullGraphMigration can have optional from_snapshot_version."""
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


# Removed test_full_graph_migration_with_description - description field was removed


# Removed test_full_graph_migration_with_metadata - metadata field was removed


# Removed test_full_graph_migration_deserialize_json_fields - deserialize_json_fields validator was removed
# Pydantic handles JSON deserialization automatically when proper types are used


def test_get_failed_features_with_retry(store):
    """Test that get_failed_features only returns features whose latest event is failure.

    If a feature fails and then succeeds on retry, it should not appear in failed_features.
    """
    from metaxy.metadata_store.system import Event, SystemTableStorage

    storage = SystemTableStorage(store)

    with store:
        migration_id = "test_migration_retry"
        project = "test"

        # Feature 1: fails then succeeds
        storage.write_event(Event.migration_started(project, migration_id))
        storage.write_event(Event.feature_started(project, migration_id, "feature_1"))
        storage.write_event(
            Event.feature_failed(project, migration_id, "feature_1", "First error")
        )
        # Retry - now succeeds
        storage.write_event(Event.feature_started(project, migration_id, "feature_1"))
        storage.write_event(
            Event.feature_completed(
                project, migration_id, "feature_1", rows_affected=100
            )
        )

        # Feature 2: fails and stays failed
        storage.write_event(Event.feature_started(project, migration_id, "feature_2"))
        storage.write_event(
            Event.feature_failed(project, migration_id, "feature_2", "Permanent error")
        )

        # Feature 3: succeeds on first try
        storage.write_event(Event.feature_started(project, migration_id, "feature_3"))
        storage.write_event(
            Event.feature_completed(
                project, migration_id, "feature_3", rows_affected=50
            )
        )

        # Get completed and failed features
        completed = storage.get_completed_features(migration_id, project)
        failed = storage.get_failed_features(migration_id, project)

        # Assertions
        assert "feature_1" in completed, (
            "Feature that succeeded after retry should be in completed"
        )
        assert "feature_3" in completed, (
            "Feature that succeeded first try should be in completed"
        )
        assert "feature_1" not in failed, (
            "Feature that succeeded after retry should NOT be in failed"
        )
        assert "feature_2" in failed, "Feature that never succeeded should be in failed"
        assert failed["feature_2"] == "Permanent error"
        assert "feature_3" not in failed, (
            "Feature that succeeded should NOT be in failed"
        )

        # Check migration summary
        summary = storage.get_migration_summary(migration_id, project)
        assert summary["status"].value == "in_progress"  # Not completed or failed
        assert len(summary["completed_features"]) == 2  # feature_1 and feature_3
        assert len(summary["failed_features"]) == 1  # only feature_2
        assert "feature_2" in summary["failed_features"]
