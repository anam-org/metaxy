from typing import TYPE_CHECKING

from pydantic import Field as PydanticField
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

if TYPE_CHECKING:
    pass


class PluginConfig(BaseSettings):
    """Configuration for Metaxy plugins"""

    model_config = SettingsConfigDict(frozen=True, extra="allow")

    enable: bool = PydanticField(
        default=False,
        description="Whether to enable the plugin.",
    )
