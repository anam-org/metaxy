"""Tests for SQLModel system table definitions."""

from __future__ import annotations

from sqlalchemy import MetaData

from metaxy.ext.sqlmodel import SQLModelPluginConfig


def test_sqlmodel_system_tables_import():
    """Test that SQLModel system tables can be imported."""
    from metaxy.ext.sqlmodel_system_tables import (
        FeatureVersionsTable,
        MigrationEventsTable,
        get_system_metadata,
        get_system_tables,
    )

    # Check that classes are importable
    assert FeatureVersionsTable is not None
    assert MigrationEventsTable is not None

    # Check that helper functions work
    tables = get_system_tables()
    assert len(tables) == 2
    assert FeatureVersionsTable in tables
    assert MigrationEventsTable in tables

    metadata = get_system_metadata()
    assert isinstance(metadata, MetaData)


def test_system_table_structure():
    """Test that system tables have correct structure."""
    from metaxy.ext.sqlmodel_system_tables import (
        FeatureVersionsTable,
        MigrationEventsTable,
    )

    from metaxy.metadata_store.system.events import (
        COL_EVENT_TYPE,
        COL_FEATURE_KEY,
        COL_PAYLOAD,
        COL_PROJECT,
        COL_TIMESTAMP,
        EVENTS_SCHEMA,
    )

    # Check FeatureVersionsTable columns
    assert hasattr(FeatureVersionsTable, "feature_key")
    assert hasattr(FeatureVersionsTable, "metaxy_feature_version")
    assert hasattr(FeatureVersionsTable, "recorded_at")
    assert hasattr(FeatureVersionsTable, "feature_spec")
    assert hasattr(FeatureVersionsTable, "feature_class_path")
    assert hasattr(FeatureVersionsTable, "metaxy_snapshot_version")

    # Check MigrationEventsTable columns match Polars EVENTS_SCHEMA exactly
    assert hasattr(MigrationEventsTable, COL_PROJECT)
    assert hasattr(MigrationEventsTable, "execution_id")
    assert hasattr(MigrationEventsTable, COL_TIMESTAMP)
    assert hasattr(MigrationEventsTable, COL_EVENT_TYPE)
    assert hasattr(MigrationEventsTable, COL_FEATURE_KEY)
    assert hasattr(MigrationEventsTable, COL_PAYLOAD)

    # Verify all EVENTS_SCHEMA columns are present in SQLModel
    for col_name in EVENTS_SCHEMA.keys():
        assert hasattr(MigrationEventsTable, col_name), f"Missing column: {col_name}"


def test_system_table_metadata():
    """Test that system tables are registered in SQLModel metadata."""
    from metaxy.ext.sqlmodel_system_tables import get_system_metadata

    metadata = get_system_metadata()

    # Check that tables are registered (SQLAlchemy converts hyphens to underscores)
    assert "metaxy_system__feature_versions" in metadata.tables
    assert "metaxy_system__events" in metadata.tables

    # Check table structure
    feature_versions = metadata.tables["metaxy_system__feature_versions"]
    assert "feature_key" in feature_versions.c
    assert "metaxy_feature_version" in feature_versions.c

    events = metadata.tables["metaxy_system__events"]
    assert "execution_id" in events.c
    assert "event_type" in events.c


def test_sqlmodel_config():
    """Test SQLModel configuration with system tables."""
    from metaxy.config import MetaxyConfig

    # Test default config
    config = MetaxyConfig()
    sqlmode_config = config.get_plugin("sqlmodel", SQLModelPluginConfig)
    assert sqlmode_config.system_tables  # Default is True
    assert not sqlmode_config.enable  # Default is False

    # Test with SQLModel enabled
    from metaxy.config import ExtConfig, SQLModelConfig

    ext_config = ExtConfig(sqlmodel=SQLModelConfig(enable=True, system_tables=True))
    config = MetaxyConfig(ext=ext_config)
    assert config.ext.sqlmodel.enable
    assert config.ext.sqlmodel.system_tables
    assert "sqlmodel" in config.plugins


def test_alembic_helpers():
    """Test Alembic integration helpers."""
    from metaxy.ext.alembic import (
        check_sqlmodel_enabled,
        get_metaxy_metadata,
    )

    # Test metadata retrieval
    metadata = get_metaxy_metadata()
    assert isinstance(metadata, MetaData)
    assert "metaxy_system__feature_versions" in metadata.tables

    # Test check_sqlmodel_enabled
    from metaxy.config import ExtConfig, MetaxyConfig, SQLModelConfig

    # Reset config first
    MetaxyConfig.reset()

    # With SQLModel disabled (default)
    assert not check_sqlmodel_enabled()

    # With SQLModel enabled
    config = MetaxyConfig(ext=ExtConfig(sqlmodel=SQLModelConfig(enable=True)))
    MetaxyConfig.set(config)
    assert check_sqlmodel_enabled()

    # Clean up
    MetaxyConfig.reset()


def test_alembic_include_tables():
    """Test including metaxy tables in existing metadata."""
    from sqlalchemy import Column, MetaData, String, Table

    from metaxy.ext.alembic import include_metaxy_tables

    # Create app metadata with a user table
    app_metadata = MetaData()
    Table(
        "users",
        app_metadata,
        Column("id", String, primary_key=True),
        Column("name", String),
    )

    # Include metaxy tables
    include_metaxy_tables(app_metadata)

    # Check that both user and metaxy tables are present
    assert "users" in app_metadata.tables
    assert "metaxy_system__feature_versions" in app_metadata.tables
    assert "metaxy_system__events" in app_metadata.tables


def test_migration_events_table_schema():
    """Test that MigrationEventsTable schema matches Polars EVENTS_SCHEMA."""
    from datetime import datetime

    from metaxy.ext.sqlmodel_system_tables import MigrationEventsTable

    from metaxy.metadata_store.system.events import EventType

    # Create an event instance
    event = MigrationEventsTable(
        project="test_project",
        execution_id="mig_001",
        timestamp=datetime(2025, 1, 13, 10, 30),
        event_type=EventType.MIGRATION_STARTED.value,
        feature_key=None,  # None for migration-level events
        payload="",  # Empty string default
    )

    assert event.project == "test_project"
    assert event.execution_id == "mig_001"
    assert event.event_type == "migration_started"
    assert event.feature_key is None
    assert event.payload == ""


def test_migration_events_feature_level():
    """Test MigrationEventsTable with feature-level event."""
    from datetime import datetime

    from metaxy.ext.sqlmodel_system_tables import MigrationEventsTable

    from metaxy.metadata_store.system.events import EventType

    # Create a feature-level event
    event = MigrationEventsTable(
        project="test_project",
        execution_id="mig_001",
        timestamp=datetime(2025, 1, 13, 10, 30),
        event_type=EventType.FEATURE_MIGRATION_COMPLETED.value,
        feature_key="feature/a",
        payload='{"type":"rows_affected","rows_affected":100}',
    )

    assert event.project == "test_project"
    assert event.execution_id == "mig_001"
    assert event.event_type == "feature_migration_completed"
    assert event.feature_key == "feature/a"
    assert event.payload == '{"type":"rows_affected","rows_affected":100}'


def test_migration_events_empty_payload():
    """Test that empty payload strings are handled correctly in SQLModel."""
    from datetime import datetime

    from metaxy.ext.sqlmodel_system_tables import MigrationEventsTable

    from metaxy.metadata_store.system.events import EventType

    # Create event with empty payload (default)
    event = MigrationEventsTable(
        project="test_project",
        execution_id="mig_001",
        timestamp=datetime(2025, 1, 13, 10, 30),
        event_type=EventType.MIGRATION_STARTED.value,
    )

    assert event.payload == ""
    assert event.feature_key is None


def test_push_graph_with_changed_feature_spec():
    """Test that pushing graph with changed feature spec works without constraint violations.

    This specifically tests the case where only metadata changes in the feature spec,
    which changes feature_spec_version but not necessarily feature_version.
    """
    from metaxy.config import MetaxyConfig
    from metaxy.metadata_store.duckdb import DuckDBMetadataStore
    from metaxy.metadata_store.system import SystemTableStorage
    from metaxy.models.feature import Feature, FeatureGraph
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.types import FeatureKey

    # Set project in config
    config = MetaxyConfig(project="test_project")
    MetaxyConfig.set(config)

    try:
        with DuckDBMetadataStore(database=":memory:") as store:
            # Create isolated graph
            graph = FeatureGraph()

            with graph.use():
                # Define feature with metadata v1
                class TestFeature(
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["test_feature"]),
                        fields=[],
                        id_columns=["sample_uid"],
                        metadata={"owner": "team_a", "version": 1},
                    ),
                ):
                    pass

                # Push graph first time
                result1 = SystemTableStorage(store).push_graph_snapshot()

                # Get the snapshot version
                snapshot_v1 = graph.snapshot_version
                assert not result1.already_pushed  # New snapshot
                assert result1.snapshot_version == snapshot_v1

            # Create new graph with changed metadata
            graph2 = FeatureGraph()

            with graph2.use():
                # Same feature but different metadata (changes feature_spec_version)
                class TestFeature(  # type: ignore
                    Feature,
                    spec=FeatureSpec(
                        key=FeatureKey(["test_feature"]),
                        fields=[],
                        id_columns=["sample_uid"],
                        metadata={"owner": "team_b", "version": 2},  # Changed metadata
                    ),
                ):
                    pass

                # Push graph second time - should work without constraint violation
                result2 = SystemTableStorage(store).push_graph_snapshot()

                snapshot_v2 = graph2.snapshot_version

                # Snapshot version is same (metadata change doesn't affect feature_version)
                assert snapshot_v1 == snapshot_v2
                assert result2.snapshot_version == snapshot_v2
                # Should be info update to existing snapshot (same fields/deps)
                assert result2.already_pushed  # Same snapshot_version
                assert (
                    "test_feature" in result2.updated_features
                )  # Feature info updated

                # Query feature_versions table to verify both specs exist
                from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

                versions_df = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
                assert versions_df is not None

                versions = versions_df.collect()

                # Should have two rows (one for each feature_spec_version)
                test_feature_versions = versions.filter(
                    versions["feature_key"] == "test_feature"
                )
                assert len(test_feature_versions) == 2

                # Verify different feature_spec_versions
                spec_versions = set(
                    test_feature_versions["metaxy_feature_spec_version"].to_list()
                )
                assert len(spec_versions) == 2  # Two different spec versions
    finally:
        # Reset config
        MetaxyConfig.reset()
