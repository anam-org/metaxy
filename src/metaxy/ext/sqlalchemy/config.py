"""Configuration for SQLAlchemy integration."""

from pydantic import Field as PydanticField
from pydantic_settings import SettingsConfigDict

from metaxy.config import PluginConfig


class SQLAlchemyConfig(PluginConfig):
    """Configuration for SQLAlchemy integration.

    This plugin provides helpers for working with SQLAlchemy metadata
    and table definitions.
    """

    model_config = SettingsConfigDict(
        env_prefix="METAXY_EXT__SQLALCHEMY_",
        extra="forbid",
    )

    inject_primary_key: bool = PydanticField(
        default=False,
        description="Automatically inject composite primary key constraints on user-defined feature tables. The key is composed of ID columns, `metaxy_created_at`, and `metaxy_data_version`.",
    )

    inject_index: bool = PydanticField(
        default=False,
        description="Automatically inject composite index on user-defined feature tables. The index covers ID columns, `metaxy_created_at`, and `metaxy_data_version`.",
    )
