"""Alembic integration helpers for metaxy.

This module provides utilities for including metaxy system tables
in Alembic migrations when using SQLModel integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from metaxy.config import MetaxyConfig
from metaxy.ext.alembic.config import AlembicPluginConfig

if TYPE_CHECKING:
    from sqlalchemy import MetaData


def get_metaxy_metadata() -> MetaData:
    """Get SQLAlchemy metadata containing metaxy system tables.

    This function returns the metadata object that should be included
    in your Alembic configuration to manage metaxy system tables.

    Returns:
        SQLAlchemy MetaData containing metaxy system table definitions

    Raises:
        ImportError: If SQLModel is not installed

    Example:
        ```py
        # In your alembic/env.py:
        from metaxy.ext.alembic import get_metaxy_metadata

        # Get system tables metadata for migrations
        target_metadata = get_metaxy_metadata()
        ```
    """
    from metaxy.ext.sqlmodel.system_tables import get_system_metadata

    return get_system_metadata()


def include_metaxy_tables(target_metadata: MetaData) -> MetaData:
    """Include metaxy system tables in existing metadata.

    This is a convenience function that adds metaxy system tables
    to your existing SQLAlchemy metadata object.

    Args:
        target_metadata: Your application's metadata object

    Returns:
        The same metadata object with metaxy tables added

    Example:
        ```py
        # In your alembic/env.py:
        from metaxy.ext.alembic import include_metaxy_tables
        from myapp.models import metadata

        # Add metaxy tables to your metadata
        target_metadata = include_metaxy_tables(metadata)
        ```
    """
    metaxy_metadata = get_metaxy_metadata()

    # Copy tables from metaxy metadata to target metadata
    for table_name, table in metaxy_metadata.tables.items():
        if table_name not in target_metadata.tables:
            # Create a new table with the same definition in target metadata
            table.to_metadata(target_metadata)

    return target_metadata


def check_sqlmodel_enabled() -> bool:
    """Check if SQLModel integration is enabled in the configuration.

    Returns:
        True if SQLModel integration is enabled, False otherwise

    Example:
        ```py
        from metaxy.ext.alembic import check_sqlmodel_enabled
        if check_sqlmodel_enabled():
            # SQLModel is enabled, include metaxy tables
            include_metaxy_tables(metadata)
        ```
    """
    try:
        config = MetaxyConfig.get()
        # Check if SQLModel plugin is enabled in the plugins list
        return "sqlmodel" in config.plugins
    except Exception:
        return False


def get_user_features_metadata(project: str | None = None) -> MetaData:
    """Get SQLAlchemy metadata for user-defined feature tables filtered by project.

    This function returns metadata containing only feature tables that belong
    to the specified project. By default, uses the project from MetaxyConfig.

    Args:
        project: Project name to filter by. If None, uses MetaxyConfig.get().project

    Returns:
        SQLAlchemy MetaData containing filtered feature table definitions

    Raises:
        ImportError: If SQLModel is not installed
        ValueError: If project filtering is enabled but no features found for project

    Example:
        ```py
        # In your alembic/env.py:
        from metaxy.ext.alembic import get_user_features_metadata
        from metaxy import init_metaxy

        # Load features
        init_metaxy()

        # Get metadata for current project only
        target_metadata = get_user_features_metadata()
        ```
    """
    from sqlalchemy import MetaData
    from sqlmodel import SQLModel

    from metaxy.ext.sqlmodel.base_feature import BaseSQLModelFeature

    config = MetaxyConfig.get()
    alembic_config = config.get_plugin("alembic", AlembicPluginConfig)

    if project is None:
        project = config.project

    # Create new metadata to hold filtered tables
    filtered_metadata = MetaData()

    # If filtering is disabled, return all SQLModel tables
    if not alembic_config.filter_by_project:
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
            if (
                hasattr(feature_cls, "__tablename__")
                and feature_cls.__tablename__ == table_name
            ):
                # Check if the feature belongs to the target project
                if hasattr(feature_cls, "project") and feature_cls.project == project:
                    should_include = True
                    break

        if should_include:
            table.to_metadata(filtered_metadata)

    return filtered_metadata


def get_store_sqlalchemy_url(store_name: str | None = None) -> str:
    """Get SQLAlchemy URL from a configured MetadataStore.

    This helper retrieves the sqlalchemy_url property from a named
    MetadataStore in the MetaxyConfig. Useful for configuring Alembic
    to use the same database as your metadata store.

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
        # In your alembic/env.py:
        from metaxy.ext.alembic import get_store_sqlalchemy_url

        # Get URL from default store
        sqlalchemy_url = get_store_sqlalchemy_url()

        # Get URL from specific store
        sqlalchemy_url = get_store_sqlalchemy_url("prod")

        # Use with Alembic config
        config.set_main_option("sqlalchemy.url", sqlalchemy_url)
        ```
    """
    config = MetaxyConfig.get()
    store = config.get_store(store_name)

    # Check if store has sqlalchemy_url property
    if not hasattr(store, "sqlalchemy_url"):
        raise AttributeError(
            f"MetadataStore '{store_name or config.store}' (type: {type(store).__name__}) "
            f"does not have a 'sqlalchemy_url' property. Only Ibis-based stores "
            f"initialized with a connection string support this feature."
        )

    url = store.sqlalchemy_url

    if not url:
        raise ValueError(
            f"MetadataStore '{store_name or config.store}' has no sqlalchemy_url available. "
            f"Ensure the store was initialized with a connection_string parameter."
        )

    return url
