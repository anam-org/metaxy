"""Configuration for Alembic integration."""

from pydantic import Field as PydanticField
from pydantic_settings import SettingsConfigDict

from metaxy.config import PluginConfig


class AlembicPluginConfig(PluginConfig):
    """Configuration for Alembic integration.

    This plugin provides helpers for integrating Metaxy with Alembic
    database migrations, including project-based filtering of feature tables.
    """

    model_config = SettingsConfigDict(env_prefix="METAXY_EXT__ALEMBIC_")

    filter_by_project: bool = PydanticField(
        default=True,
        description="Filter feature tables by current project when generating migrations. When enabled, only features belonging to the current project (from MetaxyConfig.project) are included in Alembic's target metadata.",
    )
