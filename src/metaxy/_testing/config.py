"""Test configuration classes for documentation generation testing."""

from pydantic import Field as PydanticField
from pydantic_settings import BaseSettings, SettingsConfigDict


class SamplePluginConfig(BaseSettings):
    """Sample plugin configuration for doc generation testing.

    This is a minimal config class used to test documentation generation
    without depending on the full Metaxy config structure.
    """

    model_config = SettingsConfigDict(
        env_prefix="SAMPLE_PLUGIN_",
        env_nested_delimiter="__",
        frozen=True,
    )

    enable: bool = PydanticField(
        default=False,
        description="Whether to enable the test plugin",
    )

    name: str = PydanticField(
        default="test",
        description="Name of the test plugin",
    )

    port: int = PydanticField(
        default=8080,
        description="Port number for the test service",
    )

    debug: bool = PydanticField(
        default=False,
        description="Enable debug mode",
    )

    optional_setting: str | None = PydanticField(
        default=None,
        description="Optional configuration setting",
    )
