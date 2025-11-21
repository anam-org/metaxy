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


def get_system_metadata(
    table_prefix: str = "",
    inject_primary_key: bool = False,
) -> MetaData:
    """Get SQLAlchemy metadata containing only system tables.

    Returns a new MetaData object with system table definitions.
    This can be used with database migration tools like Alembic.

    Args:
        table_prefix: Optional prefix to prepend to table names (e.g., "dev_")
        inject_primary_key: If True, include primary key constraints

    Returns:
        MetaData containing system table definitions

    Example:
        ```py
        from metaxy.ext.sqlalchemy import get_system_metadata

        # Use with Alembic target_metadata
        target_metadata = get_system_metadata()

        # With custom prefix
        target_metadata = get_system_metadata(table_prefix="dev_")
        ```
    """
    metadata = MetaData()
    create_system_tables(
        metadata, inject_primary_key=inject_primary_key, table_prefix=table_prefix
    )
    return metadata


# Helper Functions


def get_metaxy_system_metadata(
    store_name: str | None = None,
    inject_primary_key: bool = False,
) -> MetaData:
    """Get SQLAlchemy metadata containing metaxy system tables.

    This function returns the metadata object containing metaxy system tables
    which can be used with migration tools like Alembic. If a store_name is provided,
    the table_prefix from that store will be used.

    Args:
        store_name: Name of the metadata store to get table_prefix from.
                   If None, uses default store.
        inject_primary_key: If True, include primary key constraints

    Returns:
        SQLAlchemy MetaData containing metaxy system table definitions

    Example:
        ```py
        from metaxy.ext.sqlalchemy import get_metaxy_system_metadata

        # Get system tables metadata for migrations
        target_metadata = get_metaxy_system_metadata()

        # Get metadata for specific store (with its table_prefix)
        target_metadata = get_metaxy_system_metadata(store_name="prod")
        ```
    """
    config = MetaxyConfig.get()
    store = config.get_store(store_name)

    # Get table_prefix from store if available
    table_prefix = getattr(store, "table_prefix", "")

    return get_system_metadata(
        table_prefix=table_prefix,
        inject_primary_key=inject_primary_key,
    )


def add_metaxy_system_metadata(
    target_metadata: MetaData,
    store_name: str | None = None,
) -> MetaData:
    """Include metaxy system tables in existing metadata.

    This is a convenience function that adds metaxy system tables
    to your existing SQLAlchemy metadata object. If a store_name is provided,
    the table_prefix from that store will be used.

    Args:
        target_metadata: Your application's metadata object
        store_name: Name of the metadata store to get table_prefix from.
                   If None, uses default store.

    Returns:
        The same metadata object with metaxy tables added

    Example:
        ```py
        from metaxy.ext.sqlalchemy import add_metaxy_system_metadata
        from myapp.models import metadata

        # Add metaxy tables to your metadata
        target_metadata = add_metaxy_system_metadata(metadata)

        # Add tables for specific store (with its table_prefix)
        target_metadata = add_metaxy_system_metadata(metadata, store_name="prod")
        ```
    """
    metaxy_metadata = get_metaxy_system_metadata(store_name=store_name)

    # Copy tables from metaxy metadata to target metadata
    for table_name, table in metaxy_metadata.tables.items():
        if table_name not in target_metadata.tables:
            # Create a new table with the same definition in target metadata
            table.to_metadata(target_metadata)

    return target_metadata


def get_store_sqlalchemy_url(store_name: str | None = None) -> str:
    """Get SQLAlchemy URL from a configured MetadataStore.

    This helper retrieves the sqlalchemy_url property from a named
    MetadataStore in the MetaxyConfig.

    Args:
        store_name: Name of the store in MetaxyConfig.stores.
                   If None, uses the default store (MetaxyConfig.store)

    Returns:
        SQLAlchemy connection URL string

    Raises:
        ValueError: If store not found or doesn't support sqlalchemy_url
        AttributeError: If store doesn't have sqlalchemy_url property

    Example:
        ```py
        from metaxy.ext.sqlalchemy import get_store_sqlalchemy_url

        # Get URL from default store
        sqlalchemy_url = get_store_sqlalchemy_url()

        # Get URL from specific store
        sqlalchemy_url = get_store_sqlalchemy_url("prod")
        ```
    """
    config = MetaxyConfig.get()
    store = config.get_store(store_name)

    # Check if store has sqlalchemy_url property
    if not hasattr(store, "sqlalchemy_url"):
        raise AttributeError(
            f"MetadataStore '{store_name or config.store}' (type: {type(store).__name__}) "
            f"does not have a `sqlalchemy_url` property. "
        )

    url = getattr(store, "sqlalchemy_url")

    if not url:
        raise ValueError(
            f"MetadataStore '{store_name or config.store}' has an empty `sqlalchemy_url`."
        )

    return url


def get_store_metadata_and_url(
    store_name: str | None = None,
    inject_primary_key: bool = False,
) -> tuple[str, MetaData]:
    """Get both SQLAlchemy URL and system table metadata for a metadata store.

    This is a convenience function that retrieves both the connection URL
    and system table metadata for a store, with the store's table_prefix
    automatically applied to table names.

    Args:
        store_name: Name of the metadata store. If None, uses default store.
        inject_primary_key: If True, include primary key constraints

    Returns:
        Tuple of (sqlalchemy_url, system_metadata_with_table_prefix)

    Raises:
        ValueError: If store doesn't support sqlalchemy_url
        AttributeError: If store doesn't have sqlalchemy_url property

    Example:
        ```py
        from metaxy.ext.sqlalchemy import get_store_metadata_and_url

        # Get URL and metadata for default store
        url, metadata = get_store_metadata_and_url()

        # Get URL and metadata for specific store
        url, metadata = get_store_metadata_and_url("prod")

        # Use with Alembic
        from alembic import context
        url, target_metadata = get_store_metadata_and_url()
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    url = get_store_sqlalchemy_url(store_name)
    metadata = get_metaxy_system_metadata(
        store_name, inject_primary_key=inject_primary_key
    )
    return url, metadata


def get_features_sqlalchemy_metadata(
    project: str | None = None,
    filter_by_project: bool = True,
) -> MetaData:
    """Get SQLAlchemy metadata for user-defined feature tables filtered by project.

    This function returns metadata containing only feature tables that belong
    to the specified project. By default, uses the project from MetaxyConfig
    and filters by that project.

    Note:
        This function requires SQLModel to be installed and looks for tables
        registered in SQLModel.metadata.

    Args:
        project: Project name to filter by. If None, uses MetaxyConfig.get().project
        filter_by_project: If True, only include features for the specified project.
                          If False, include all SQLModel tables.

    Returns:
        SQLAlchemy MetaData containing filtered feature table definitions

    Raises:
        ImportError: If SQLModel is not installed

    Example:
        ```py
        from metaxy.ext.sqlalchemy import get_features_sqlalchemy_metadata
        from metaxy import init_metaxy

        # Load features
        init_metaxy()

        # Get metadata for current project only (default)
        target_metadata = get_features_sqlalchemy_metadata()

        # Get metadata for all projects
        target_metadata = get_features_sqlalchemy_metadata(filter_by_project=False)
        ```
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
        # Find the corresponding Feature class
        # Iterate through all registered SQLModelFeature classes
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


def get_features_store_metadata_and_url(
    store_name: str | None = None,
    project: str | None = None,
    filter_by_project: bool = True,
) -> tuple[str, MetaData]:
    """Get both SQLAlchemy URL and feature metadata for a metadata store.

    This is a convenience function that retrieves both the connection URL
    and feature table metadata for a store. This makes it easy to configure
    migration tools like Alembic.

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
        ValueError: If store doesn't support sqlalchemy_url
        AttributeError: If store doesn't have sqlalchemy_url property
        ImportError: If SQLModel is not installed

    Example:
        ```py
        from metaxy.ext.sqlalchemy import get_features_store_metadata_and_url
        from metaxy import init_metaxy

        # Load features
        init_metaxy()

        # Get URL and metadata for default store
        url, metadata = get_features_store_metadata_and_url()

        # Use with Alembic
        from alembic import context
        url, target_metadata = get_features_store_metadata_and_url()
        context.configure(url=url, target_metadata=target_metadata)
        ```
    """
    url = get_store_sqlalchemy_url(store_name)
    metadata = get_features_sqlalchemy_metadata(project, filter_by_project)
    return url, metadata
