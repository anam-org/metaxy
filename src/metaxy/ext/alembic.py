"""Alembic integration helpers for metaxy.

This module provides utilities for including metaxy system tables
in Alembic migrations when using SQLModel integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        from sqlalchemy import MetaData

        # Combine your app metadata with metaxy metadata
        target_metadata = MetaData()
        target_metadata.reflect(get_metaxy_metadata())
        target_metadata.reflect(my_app.metadata)
        ```
    """
    from metaxy.ext.sqlmodel_system_tables import get_system_metadata

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
    from metaxy.ext.sqlmodel_system_tables import get_system_metadata

    metaxy_metadata = get_system_metadata()

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
        from metaxy.config import MetaxyConfig

        config = MetaxyConfig.get()
        # Check if SQLModel plugin is enabled in the plugins list
        return "sqlmodel" in config.plugins
    except Exception:
        return False
