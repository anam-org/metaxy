"""SQLModel definitions for metaxy system tables.

This module provides SQLModel table classes that extend the Pydantic models
from system_tables.py with SQLAlchemy-specific configurations. These are used
primarily for Alembic migrations when the SQLModel integration is enabled.

The actual data flow remains unchanged (using Polars DataFrames).
These models are only for schema definition and migrations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Index
from sqlmodel import Field, SQLModel

# Import the Pydantic models and constants from the core module
from metaxy.metadata_store.system import (
    EVENTS_KEY,
    FEATURE_VERSIONS_KEY,
    METAXY_SYSTEM_KEY_PREFIX,
    FeatureVersionsModel,
)
from metaxy.models.constants import (
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_FULL_DEFINITION_VERSION,
    METAXY_SNAPSHOT_VERSION,
)

# System tables that metaxy uses internally
SYSTEM_TABLES: list[str] = [
    f"{METAXY_SYSTEM_KEY_PREFIX}__feature_versions",
    f"{METAXY_SYSTEM_KEY_PREFIX}__events",
]


class FeatureVersionsTable(FeatureVersionsModel, SQLModel, table=True):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """SQLModel definition for the feature_versions system table.

    Extends FeatureVersionsModel from system_tables.py with SQLModel-specific
    configuration for database table creation and Alembic migrations.
    """

    __tablename__: str = FEATURE_VERSIONS_KEY.table_name  # pyright: ignore[reportIncompatibleVariableOverride]

    # Override fields that need SQLAlchemy-specific configuration (primary keys, indexes, column names)
    # Composite primary key: (project, feature_key, feature_spec_version)
    project: str = Field(primary_key=True, index=True)
    feature_key: str = Field(primary_key=True)
    metaxy_snapshot_version: str = Field(
        index=True,
        sa_column_kwargs={"name": METAXY_SNAPSHOT_VERSION},
    )
    metaxy_feature_version: str = Field(
        index=True,
        sa_column_kwargs={"name": METAXY_FEATURE_VERSION},
    )
    metaxy_feature_spec_version: str = Field(
        primary_key=True,
        sa_column_kwargs={"name": METAXY_FEATURE_SPEC_VERSION},
    )
    metaxy_full_definition_version: str = Field(
        index=True,
        sa_column_kwargs={"name": METAXY_FULL_DEFINITION_VERSION},
    )

    # Additional indexes for common queries
    __table_args__ = (
        Index(
            f"idx_{FEATURE_VERSIONS_KEY.table_name}_lookup",
            "project",
            "feature_key",
            METAXY_FEATURE_VERSION,
        ),
    )


class MigrationEventsTable(SQLModel, table=True):
    """SQLModel definition for the events system table.

    This is a pure schema definition for the database table structure.
    Application code uses typed event models (Event with classmethods)
    which serialize to this schema via to_polars().

    Note: All columns match the Polars EVENTS_SCHEMA exactly.
    - event_type: stored as string (not enum) for maximum backend compatibility
    - payload: stored as JSON string (not JSON column) for consistency with Polars
    - execution_id: generic name in storage (migration_id in CLI is user-facing)
    """

    __tablename__: str = EVENTS_KEY.table_name  # pyright: ignore[reportIncompatibleVariableOverride]

    # Composite primary key matching Polars append-only storage
    project: str = Field(primary_key=True, index=True)
    execution_id: str = Field(primary_key=True, index=True)
    timestamp: datetime = Field(primary_key=True)

    # Event fields
    event_type: str = Field(index=True)  # Stored as string for backend compatibility
    feature_key: str | None = Field(
        default=None, nullable=True
    )  # None for execution-level events
    payload: str = Field(default="")  # JSON string for consistency with Polars

    # Additional indexes for common queries
    __table_args__ = (
        Index(
            f"idx_{EVENTS_KEY.table_name}_lookup",
            "project",
            "execution_id",
            "event_type",
        ),
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
        ```py
        from metaxy.ext.sqlmodel_system_tables import get_system_metadata
        metadata = get_system_metadata()
        # Use with Alembic target_metadata
        ```
    """
    # All tables are automatically registered in SQLModel.metadata
    # when their classes are defined with table=True
    return SQLModel.metadata
