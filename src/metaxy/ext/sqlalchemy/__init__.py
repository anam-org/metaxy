"""SQLAlchemy integration for metaxy.

This module provides SQLAlchemy table definitions and helpers for metaxy.
These can be used with migration tools like Alembic.

The main functions return tuples of (sqlalchemy_url, metadata) for easy
integration with migration tools:

- `get_system_sqla_metadata_for_store`: Get URL and system table metadata for a store
- `get_feature_sqla_metadata_for_store`: Get URL and feature table metadata for a store
"""

from metaxy.ext.sqlalchemy.config import SQLAlchemyConfig
from metaxy.ext.sqlalchemy.plugin import (
    get_feature_sqla_metadata_for_store,
    get_system_sqla_metadata_for_store,
)

__all__ = [
    "SQLAlchemyConfig",
    "get_system_sqla_metadata_for_store",
    "get_feature_sqla_metadata_for_store",
]
