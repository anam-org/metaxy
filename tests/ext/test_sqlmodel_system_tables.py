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

    # Check FeatureVersionsTable columns
    assert hasattr(FeatureVersionsTable, "feature_key")
    assert hasattr(FeatureVersionsTable, "metaxy_feature_version")
    assert hasattr(FeatureVersionsTable, "recorded_at")
    assert hasattr(FeatureVersionsTable, "feature_spec")
    assert hasattr(FeatureVersionsTable, "feature_class_path")
    assert hasattr(FeatureVersionsTable, "metaxy_snapshot_version")

    # Check MigrationEventsTable columns
    assert hasattr(MigrationEventsTable, "id")
    assert hasattr(MigrationEventsTable, "migration_id")
    assert hasattr(MigrationEventsTable, "event_type")
    assert hasattr(MigrationEventsTable, "timestamp")
    assert hasattr(MigrationEventsTable, "feature_key")
    assert hasattr(MigrationEventsTable, "rows_affected")
    assert hasattr(MigrationEventsTable, "error_message")


def test_system_table_metadata():
    """Test that system tables are registered in SQLModel metadata."""
    from metaxy.ext.sqlmodel_system_tables import get_system_metadata

    metadata = get_system_metadata()

    # Check that tables are registered
    assert "metaxy-system__feature_versions" in metadata.tables
    assert "metaxy-system__migration_events" in metadata.tables

    # Check table structure
    feature_versions = metadata.tables["metaxy-system__feature_versions"]
    assert "feature_key" in feature_versions.c
    assert "metaxy_feature_version" in feature_versions.c

    migration_events = metadata.tables["metaxy-system__migration_events"]
    assert "migration_id" in migration_events.c
    assert "event_type" in migration_events.c


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
    assert "metaxy-system__feature_versions" in metadata.tables

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
    assert "metaxy-system__feature_versions" in app_metadata.tables
    assert "metaxy-system__migration_events" in app_metadata.tables
