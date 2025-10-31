"""SQLModel definitions for metaxy system tables.

This module provides SQLModel table classes that mirror the Polars schemas
from system_tables.py. These are used primarily for Alembic migrations
when the SQLModel integration is enabled.

The actual data flow remains unchanged (using Polars DataFrames).
These models are only for schema definition and migrations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Index
from sqlmodel import Field, SQLModel

# Import the namespace constant from the core module
from metaxy.metadata_store.system_tables import SYSTEM_NAMESPACE

# System tables that metaxy uses internally
SYSTEM_TABLES: list[str] = [
    f"{SYSTEM_NAMESPACE}__feature_versions",
    f"{SYSTEM_NAMESPACE}__migration_events",
]


class FeatureVersionsTable(SQLModel, table=True):
    """SQLModel definition for the feature_versions system table.

    This table records when feature specifications are pushed to production,
    tracking the evolution of feature definitions over time.
    """

    __tablename__: str = f"{SYSTEM_NAMESPACE}__feature_versions"  # pyright: ignore[reportIncompatibleVariableOverride]

    # Composite primary key: project + feature_key + snapshot_version
    project: str = Field(primary_key=True, index=True)
    feature_key: str = Field(primary_key=True)
    snapshot_version: str = Field(primary_key=True)

    # Version and timestamp
    feature_version: str = Field(index=True)
    feature_spec_version: str = Field(index=True)  # Hash of complete BaseFeatureSpec
    recorded_at: datetime = Field(index=True)

    # Serialized feature specification and class path
    feature_spec: str  # JSON string
    feature_class_path: str

    # Additional indexes for common queries
    __table_args__ = (
        Index(
            "idx_feature_versions_lookup", "project", "feature_key", "feature_version"
        ),
    )


class MigrationEventsTable(SQLModel, table=True):
    """SQLModel definition for the migration_events system table.

    This table stores append-only events tracking migration execution,
    enabling recovery from partial failures and progress monitoring.
    """

    __tablename__: str = f"{SYSTEM_NAMESPACE}__migration_events"  # pyright: ignore[reportIncompatibleVariableOverride]

    # Auto-incrementing ID for append-only events
    id: int | None = Field(default=None, primary_key=True)

    # Event fields
    project: str = Field(index=True)
    migration_id: str = Field(index=True)
    event_type: str = (
        Field()
    )  # "started", "feature_started", "feature_completed", "completed", "failed"
    timestamp: datetime = Field(index=True)
    feature_key: str = Field(default="")  # Empty for migration-level events
    rows_affected: int = Field(default=0)
    error_message: str = Field(default="")  # Empty if no error

    # Additional indexes for common queries
    __table_args__ = (
        Index("idx_migration_events_lookup", "project", "migration_id", "event_type"),
    )


def get_system_tables() -> list[type[SQLModel]]:
    """Get all system table SQLModel classes.

    Returns:
        List of SQLModel table classes
    """
    return [FeatureVersionsTable, MigrationEventsTable]


def get_system_metadata():
    """Get SQLAlchemy metadata containing all system tables.

    This function returns the metadata object that contains all
    system table definitions, which can be used with Alembic
    for migration generation.

    Returns:
        SQLModel.metadata containing all system table definitions

    Example:
        >>> from metaxy.ext.sqlmodel_system_tables import get_system_metadata
        >>> metadata = get_system_metadata()
        >>> # Use with Alembic target_metadata
    """
    # All tables are automatically registered in SQLModel.metadata
    # when their classes are defined with table=True
    return SQLModel.metadata
