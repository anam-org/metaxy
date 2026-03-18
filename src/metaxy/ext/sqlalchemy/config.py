"""Configuration for SQLAlchemy integration."""

from pydantic_settings import SettingsConfigDict

from metaxy._decorators import public
from metaxy.config import PluginConfig


@public
class SQLAlchemyConfig(PluginConfig):
    """Configuration for SQLAlchemy integration.

    This plugin provides helpers for working with SQLAlchemy metadata
    and table definitions.
    """

    model_config = SettingsConfigDict(
        env_prefix="METAXY_EXT__SQLALCHEMY_",
        extra="forbid",
    )
