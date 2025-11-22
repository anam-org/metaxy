"""SQLAlchemy integration plugin for metaxy.

This module provides SQLAlchemy Table definitions and helpers for metaxy system tables
and user-defined feature tables. These can be used with migration tools like Alembic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, Index, MetaData, String, Table

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.system import EVENTS_KEY, FEATURE_VERSIONS_KEY
from metaxy.models.constants import (
    METAXY_FEATURE_SPEC_VERSION,
    METAXY_FEATURE_VERSION,
    METAXY_FULL_DEFINITION_VERSION,
    METAXY_SNAPSHOT_VERSION,
)

if TYPE_CHECKING:
    pass


# System Tables


def create_system_tables(
    metadata: MetaData,
    inject_primary_key: bool = False,
    table_prefix: str = "",
) -> tuple[Table, Table]:
    """Create system table definitions in the given metadata.

    Args:
        metadata: SQLAlchemy MetaData object to add tables to
        inject_primary_key: If True, include primary key constraints.
                           If False, create tables without primary keys (default).
        table_prefix: Optional prefix to prepend to table names (e.g., "dev_")

    Returns:
        Tuple of (feature_versions_table, events_table)
    """
    feature_versions_name = (
        f"{table_prefix}{FEATURE_VERSIONS_KEY.table_name}"
        if table_prefix
        else FEATURE_VERSIONS_KEY.table_name
    )

    feature_versions_table = Table(
        feature_versions_name,
        metadata,
        # Composite primary key
        Column("project", String, primary_key=inject_primary_key, index=True),
        Column("feature_key", String, primary_key=inject_primary_key, index=True),
        Column(
            METAXY_FEATURE_SPEC_VERSION,
            String,
            primary_key=inject_primary_key,
        ),
        # Versioning columns
        Column(METAXY_FEATURE_VERSION, String, index=True),
        Column(METAXY_FULL_DEFINITION_VERSION, String, index=True),
        Column(METAXY_SNAPSHOT_VERSION, String, index=True),
        # Metadata columns
        Column("recorded_at", DateTime, index=True),
        Column("feature_schema", String),  # JSON string
        Column("tags", String, default="{}"),  # JSON string
        # Additional indexes
        Index(
            f"idx_{feature_versions_name}_lookup",
            "project",
            "feature_key",
            METAXY_FEATURE_VERSION,
        ),
    )

    events_name = (
        f"{table_prefix}{EVENTS_KEY.table_name}"
        if table_prefix
        else EVENTS_KEY.table_name
    )

    events_table = Table(
        events_name,
        metadata,
        # Composite primary key matching Polars append-only storage
        Column("project", String, primary_key=inject_primary_key, index=True),
        Column("execution_id", String, primary_key=inject_primary_key, index=True),
        Column("timestamp", DateTime, primary_key=inject_primary_key),
        # Event fields
        Column("event_type", String, index=True),
        Column("feature_key", String, nullable=True, index=True),
        Column("payload", String, default=""),  # JSON string
        # Additional indexes
        Index(
            f"idx_{events_name}_lookup",
            "project",
            "execution_id",
            "event_type",
        ),
    )

    return feature_versions_table, events_table


# Internal helper for getting store's table_prefix
def _get_table_prefix(store_name: str | None = None) -> str:
    """Get table_prefix from a metadata store.

    Args:
        store_name: Name of the metadata store. If None, uses default store.

    Returns:
        Table prefix string (empty string if not set)
    """
    config = MetaxyConfig.get()
    store = config.get_store(store_name)
    return getattr(store, "table_prefix", "")


def _get_store_sqlalchemy_url(store_name: str | None = None) -> str:
    """Get SQLAlchemy URL from a configured MetadataStore.

    Args:
        store_name: Name of the store. If None, uses default store.

    Returns:
        SQLAlchemy connection URL string

    Raises:
        AttributeError: If store doesn't have sqlalchemy_url property
        ValueError: If sqlalchemy_url is empty
    """
    config = MetaxyConfig.get()
    store = config.get_store(store_name)

    if not hasattr(store, "sqlalchemy_url"):
        raise AttributeError(
            f"MetadataStore '{store_name or config.store}' (type: {type(store).__name__}) "
            f"does not have a `sqlalchemy_url` property."
        )

    url = getattr(store, "sqlalchemy_url")

    if not url:
        raise ValueError(
            f"MetadataStore '{store_name or config.store}' has an empty `sqlalchemy_url`."
        )

    return url


def _get_system_metadata(
    table_prefix: str = "",
    inject_primary_key: bool = False,
) -> MetaData:
    """Create SQLAlchemy metadata containing system tables.

    Args:
        table_prefix: Optional prefix to prepend to table names
        inject_primary_key: If True, include primary key constraints

    Returns:
        MetaData containing system table definitions
    """
    metadata = MetaData()
    create_system_tables(
        metadata, inject_primary_key=inject_primary_key, table_prefix=table_prefix
    )
    return metadata


def get_system_sqla_metadata_for_store(
    store_name: str | None = None,
    inject_primary_key: bool = False,
) -> tuple[str, MetaData]:
    """Get SQLAlchemy URL and system table metadata for a metadata store.

    This function retrieves both the connection URL and system table metadata
    for a store, with the store's table_prefix automatically applied to table names.

    Args:
        store_name: Name of the metadata store. If None, uses default store.
        inject_primary_key: If True, include primary key constraints.

    Returns:
        Tuple of (sqlalchemy_url, system_metadata)

    Raises:
        AttributeError: If store doesn't have sqlalchemy_url property
        ValueError: If store's sqlalchemy_url is empty

    Example:

        ```py
        from metaxy.ext.sqlalchemy import get_system_sqla_metadata_for_store

        # Get URL and metadata for default store
        url, metadata = get_system_sqla_metadata_for_store()

        # Get URL and metadata for specific store
        url, metadata = get_system_sqla_metadata_for_store("prod")

        # Use with Alembic env.py
        from alembic import context
        url, target_metadata = get_system_sqla_metadata_for_store()
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    url = _get_store_sqlalchemy_url(store_name)
    table_prefix = _get_table_prefix(store_name)
    metadata = _get_system_metadata(
        table_prefix=table_prefix, inject_primary_key=inject_primary_key
    )
    return url, metadata


def _get_features_metadata(
    project: str | None = None,
    filter_by_project: bool = True,
) -> MetaData:
    """Get SQLAlchemy metadata for user-defined feature tables.

    Args:
        project: Project name to filter by. If None, uses MetaxyConfig.get().project
        filter_by_project: If True, only include features for the specified project.

    Returns:
        SQLAlchemy MetaData containing feature table definitions

    Raises:
        ImportError: If SQLModel is not installed
    """
    from sqlalchemy import MetaData
    from sqlmodel import SQLModel

    from metaxy.ext.sqlmodel.plugin import BaseSQLModelFeature

    config = MetaxyConfig.get()

    if project is None:
        project = config.project

    # Create new metadata to hold filtered tables
    filtered_metadata = MetaData()

    # If filtering is disabled, return all SQLModel tables
    if not filter_by_project:
        for table_name, table in SQLModel.metadata.tables.items():
            table.to_metadata(filtered_metadata)
        return filtered_metadata

    # Filter tables by project using Feature.project class attribute
    for table_name, table in SQLModel.metadata.tables.items():
        should_include = False

        for feature_cls in BaseSQLModelFeature.__subclasses__():
            # Check if this feature class owns this table
            if hasattr(feature_cls, "__tablename__"):
                tablename = getattr(feature_cls, "__tablename__", None)
                if tablename == table_name:
                    # Check if the feature belongs to the target project
                    if hasattr(feature_cls, "project"):
                        feature_project = getattr(feature_cls, "project", None)
                        if feature_project == project:
                            should_include = True
                            break

        if should_include:
            table.to_metadata(filtered_metadata)

    return filtered_metadata


def get_feature_sqla_metadata_for_store(
    store_name: str | None = None,
    project: str | None = None,
    filter_by_project: bool = True,
) -> tuple[str, MetaData]:
    """Get SQLAlchemy URL and feature table metadata for a metadata store.

    This function retrieves both the connection URL and feature table metadata
    for a store. By default, filters tables to include only features belonging
    to the specified project.

    Note:
        This function requires SQLModel to be installed and looks for tables
        registered in SQLModel.metadata.

    Args:
        store_name: Name of the metadata store. If None, uses default store.
        project: Project name to filter by. If None, uses MetaxyConfig.get().project
        filter_by_project: If True, only include features for the specified project.
                          If False, include all SQLModel tables.

    Returns:
        Tuple of (sqlalchemy_url, feature_metadata)

    Raises:
        AttributeError: If store doesn't have sqlalchemy_url property
        ValueError: If store's sqlalchemy_url is empty
        ImportError: If SQLModel is not installed

    Example:

        ```py
        from metaxy.ext.sqlalchemy import get_feature_sqla_metadata_for_store
        from metaxy import init_metaxy

        # Load features first
        init_metaxy()

        # Get URL and metadata for default store
        url, metadata = get_feature_sqla_metadata_for_store()

        # Get URL and metadata for specific store and project
        url, metadata = get_feature_sqla_metadata_for_store("prod", "my_project")

        # Use with Alembic env.py
        from alembic import context
        url, target_metadata = get_feature_sqla_metadata_for_store()
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    url = _get_store_sqlalchemy_url(store_name)
    metadata = _get_features_metadata(project, filter_by_project)
    return url, metadata
