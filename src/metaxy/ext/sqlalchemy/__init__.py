"""SQLAlchemy integration for metaxy.

This module provides pure SQLAlchemy table definitions and helpers for metaxy,
independent of any ORM like SQLModel. These can be used with migration tools
like Alembic.
"""

from metaxy.ext.sqlalchemy.config import SQLAlchemyConfig
from metaxy.ext.sqlalchemy.plugin import (
    add_metaxy_system_metadata,
    get_features_sqlalchemy_metadata,
    get_features_store_metadata_and_url,
    get_metaxy_system_metadata,
    get_store_metadata_and_url,
    get_store_sqlalchemy_url,
    get_system_metadata,
)

__all__ = [
    "SQLAlchemyConfig",
    "get_system_metadata",
    "get_metaxy_system_metadata",
    "add_metaxy_system_metadata",
    "get_store_sqlalchemy_url",
    "get_store_metadata_and_url",
    "get_features_sqlalchemy_metadata",
    "get_features_store_metadata_and_url",
]
