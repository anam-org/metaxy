"""CLI application context for sharing state across commands."""

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaxy.config import MetaxyConfig
    from metaxy.metadata_store.base import MetadataStore


class AppContext:
    """CLI application context.

    Stores the config initialized by the meta app launcher.
    Commands access it via get_config() and instantiate stores as needed.
    """

    def __init__(self, config: "MetaxyConfig"):
        """Initialize context.

        Args:
            config: Metaxy configuration
        """
        self.config = config


# Context variable for storing the app context
_app_context: ContextVar[AppContext | None] = ContextVar("_app_context", default=None)


def set_config(config: "MetaxyConfig") -> None:
    """Set the config in CLI context.

    Args:
        config: Metaxy configuration
    """
    _app_context.set(AppContext(config))


def get_config() -> "MetaxyConfig":
    """Get the Metaxy config from context.

    Returns:
        MetaxyConfig instance

    Raises:
        RuntimeError: If context not initialized
    """
    ctx = _app_context.get()
    if ctx is None:
        raise RuntimeError(
            "CLI context not initialized. Config must be set by meta app."
        )

    return ctx.config


def get_store(name: str | None = None) -> "MetadataStore":
    """Get and open a metadata store from config.

    Opens the store for the duration of the command.
    Store is retrieved from config context.

    Returns:
        Opened metadata store instance (within context manager)

    Raises:
        RuntimeError: If context not initialized
    """
    config = get_config()
    return config.get_store(name)
