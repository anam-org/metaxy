"""Tests for SQLModel system table definitions."""

from __future__ import annotations

from sqlalchemy import MetaData


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
    assert config.ext.sqlmodel.system_tables  # Default is True
    assert not config.ext.sqlmodel.enable  # Default is False

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
