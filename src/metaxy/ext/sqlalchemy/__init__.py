"""SQLAlchemy integration for metaxy.

This module provides SQLAlchemy table definitions and helpers for metaxy.
These can be used with migration tools like Alembic.

The main functions return tuples of (sqlalchemy_url, metadata) for easy
integration with migration tools:

- `get_system_slqa_metadata`: Get URL and system table metadata for a store
- `filter_feature_sqla_metadata`: Get URL and feature table metadata for a store
"""

from metaxy.ext.sqlalchemy.config import SQLAlchemyConfig
from metaxy.ext.sqlalchemy.plugin import (
    filter_feature_sqla_metadata,
    get_system_slqa_metadata,
)

__all__ = [
    "SQLAlchemyConfig",
    "get_system_slqa_metadata",
    "filter_feature_sqla_metadata",
]
