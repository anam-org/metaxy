"""Configuration for SQLModel integration."""

from pydantic import Field as PydanticField
from pydantic_settings import SettingsConfigDict

from metaxy.config import PluginConfig


class SQLModelPluginConfig(PluginConfig):
    """Configuration for SQLModel integration.

    This plugin enhances SQLModel-based features with automatic table name
    inference and optional primary key injection.
    """

    model_config = SettingsConfigDict(
        env_prefix="METAXY_EXT__SQLMODEL_",
        extra="forbid",
    )

    inject_primary_key: bool = PydanticField(
        default=False,
        description="Automatically inject composite primary key constraints on SQLModel tables. The key is composed of ID columns, `metaxy_created_at`, and `metaxy_data_version`.",
    )

    inject_index: bool = PydanticField(
        default=False,
        description="Automatically inject composite index on SQLModel tables. The index covers ID columns, `metaxy_created_at`, and `metaxy_data_version`.",
    )
