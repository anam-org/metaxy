import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import tomli_w
from pydantic import Field as PydanticField
from pydantic import PrivateAttr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from typing_extensions import Self

from metaxy._decorators import public
from metaxy.config.metaxy_source import MetaxyTomlSource, discover_config_with_parents
from metaxy.config.models.plugin_config import PluginConfig
from metaxy.config.models.store_config import StoreConfig
from metaxy.config.utils import _collect_dict_keys, _remove_none_values
from metaxy.models.feature_selection import FeatureSelection

if TYPE_CHECKING:
    from metaxy.metadata_store.base import (
        MetadataStore,
    )


BUILTIN_PLUGINS = {}


PluginConfigT = TypeVar("PluginConfigT", bound=PluginConfig)
StoreTypeT = TypeVar("StoreTypeT", bound="MetadataStore")

# Global config visible to all threads, set via MetaxyConfig.set() / .load().
# ContextVar overlay for per-context overrides via MetaxyConfig.use().
# get() checks the ContextVar first, falling back to the global.
_global_config: "MetaxyConfig | None" = None
_config_override: ContextVar["MetaxyConfig | None"] = ContextVar("_config_override", default=None)

# Used by load() to pass the config file path into settings_customise_sources
# without monkey-patching the classmethod.
_toml_file_override: ContextVar[Path | None] = ContextVar("_toml_file_override")


@public
class InvalidConfigError(Exception):
    """Raised when Metaxy configuration is invalid.

    This error includes helpful context about where the configuration was loaded from
    and how environment variables can affect configuration.
    """

    def __init__(
        self,
        message: str,
        *,
        config_file: Path | None = None,
    ):
        self.config_file = config_file
        self.base_message = message

        # Build the full error message with context
        parts = [message]

        if config_file:
            parts.append(f"Config file: {config_file}")

        parts.append("Note: METAXY_* environment variables can override config file settings ")

        super().__init__("\n".join(parts))

    @classmethod
    def from_config(cls, config: "MetaxyConfig", message: str) -> "InvalidConfigError":
        """Create an InvalidConfigError from a MetaxyConfig instance.

        Args:
            config: The MetaxyConfig instance that has the invalid configuration.
            message: The error message describing what's wrong.

        Returns:
            An InvalidConfigError with context from the config.
        """
        return cls(message, config_file=config._config_file)


@public
class MetaxyConfig(BaseSettings):
    """Main Metaxy configuration.

    Loads from (in order of precedence):

    1. Init arguments

    2. Environment variables (METAXY_*)

    3. Config file (`metaxy.toml` or `[tool.metaxy]` in `pyproject.toml` )

    Environment variables can be templated with `${MY_VAR:-default}` syntax.

    Example: Accessing current configuration
        <!-- skip next -->
        ```py
        config = MetaxyConfig.load()
        ```

    Example: Getting a configured metadata store
        ```py
        store = config.get_store("prod")
        ```
    """

    model_config = SettingsConfigDict(
        env_prefix="METAXY_",
        env_nested_delimiter="__",
        frozen=True,  # Make the config immutable
    )

    store: str = PydanticField(
        default="dev",
        description="Default metadata store to use",
    )

    stores: dict[str, StoreConfig] = PydanticField(
        default_factory=dict,
        description="Named store configurations",
    )

    @model_validator(mode="before")
    @classmethod
    def _filter_incomplete_stores(cls, data: Any) -> Any:
        """Filter out incomplete store configs (e.g. from random environment variables).

        When env vars like METAXY_STORES__PROD__CONFIG__CONNECTION_STRING are set
        without METAXY_STORES__PROD__TYPE, pydantic-settings creates a partial dict
        that would fail validation. This validator removes such incomplete entries
        and emits a warning.
        """
        if not isinstance(data, dict) or "stores" not in data:
            return data

        stores = data["stores"]

        if not isinstance(stores, dict):
            return data

        complete_stores = {}

        for name, config in stores.items():
            is_complete = isinstance(config, StoreConfig) or (isinstance(config, dict) and ("type" in config))
            if is_complete:
                complete_stores[name] = config
            else:
                fields = _collect_dict_keys(config) if isinstance(config, dict) else []
                fields_hint = f" (has fields: {', '.join(fields)})" if fields else ""
                warnings.warn(
                    f"Ignoring incomplete store config '{name}': missing required 'type' field{fields_hint}. "
                    f"This is typically caused by environment variables.",
                    UserWarning,
                    stacklevel=2,
                )

        data["stores"] = complete_stores

        return data

    extend: str | None = PydanticField(
        default=None, description="A relative or absolute path to a Metaxy configuration file to inherit settings from."
    )

    entrypoints: list[str] = PydanticField(
        default_factory=list,
        description="List of Python module paths to load for feature discovery",
    )

    theme: str = PydanticField(
        default="default",
        description="Graph rendering theme for CLI visualization",
    )

    ext: dict[str, PluginConfig] = PydanticField(
        default_factory=dict,
        description="Configuration for Metaxy integrations with third-party tools",
        frozen=False,
    )

    hash_truncation_length: int = PydanticField(default=8, description="Truncate hash values to this length.", ge=8)

    enable_map_datatype: bool = PydanticField(
        default=False,
        description="Preserve [`Map` datatype](/guide/concepts/metadata-stores.md/#map-datatype) across Metaxy operations. Requires `polars-map` to be installed. Experimental.",
    )

    auto_create_tables: bool = PydanticField(
        default=False,
        description="Auto-create tables when opening stores. It is not advised to enable this setting in production.",
    )

    project: str | None = PydanticField(
        default=None,
        description="[Project](/guide/concepts/projects.md) name. Used to scope operations to enable multiple independent projects in a shared metadata store. Does not modify feature keys or table names. Project names must be valid alphanumeric strings with dashes, underscores, and cannot contain forward slashes (`/`) or double underscores (`__`)",
    )

    locked: bool | None = PydanticField(
        default=None,
        description="Whether to raise an error if an external feature doesn't have a matching feature version when [syncing external features][metaxy.sync_external_features] from the metadata store.",
    )

    sync: bool = PydanticField(
        default=True,
        description="Whether to automatically [sync external feature definitions][metaxy.sync_external_features] from the metadata during some operations. It's recommended to keep this enabled as it ensures versioning correctness for external feature definitions with a negligible performance impact.",
    )

    extra_features: list[FeatureSelection] = PydanticField(
        default_factory=list,
        description="Extra features to load from the metadata store when calling [`sync_external_features`][metaxy.sync_external_features]. Each entry is a [`FeatureSelection`][metaxy.FeatureSelection]. All entries are combined together. Learn more [here](/guide/concepts/definitions/external-features.md/#loading-extra-features).",
    )

    metaxy_lock_path: str | None = PydanticField(
        default=None,
        description="Relative or absolute path to the lock file, resolved from the config file's location. Defaults to `metaxy.lock` next to the config file.",
    )

    # Private attribute to track which config file was used (set by load())
    _config_file: Path | None = PrivateAttr(default=None)

    @property
    def config_file(self) -> Path | None:
        """The config file path used to load this configuration.

        Returns `None` if the config was created directly (not via [`MetaxyConfig.load`][metaxy.MetaxyConfig.load]).
        """
        return self._config_file

    @cached_property
    def lock_file(self) -> Path | None:
        """The resolved lock file path.

        When ``metaxy_lock_path`` is set, returns the resolved path (absolute
        paths are returned directly, relative paths are resolved from the
        config file's directory).

        When ``metaxy_lock_path`` is ``None`` (the default), checks for
        ``metaxy.lock`` next to the config file.

        Returns ``None`` if no config file is set and the path is relative.
        """
        if self.metaxy_lock_path is not None:
            lock_path = Path(self.metaxy_lock_path)
            if lock_path.is_absolute():
                return lock_path
            if self._config_file is None:
                return None
            return self._config_file.parent / lock_path

        if self._config_file is None:
            return None
        return self._config_file.parent / "metaxy.lock"

    def _load_plugins(self) -> None:
        """Load enabled plugins. Must be called after config is set."""
        for name, module in BUILTIN_PLUGINS.items():
            if name in self.ext and self.ext[name].enable:
                try:
                    __import__(module)
                except Exception as e:
                    raise InvalidConfigError.from_config(
                        self,
                        f"Failed to load Metaxy plugin '{name}' (defined in \"ext\" config field): {e}",
                    ) from e

    @field_validator("project")
    @classmethod
    def validate_project(cls, v: str | None) -> str | None:
        """Validate project name follows naming rules."""
        if v is None:
            return None
        if not v:
            raise ValueError("project name cannot be empty")
        if "/" in v:
            raise ValueError(
                f"project name '{v}' cannot contain forward slashes (/). "
                f"Forward slashes are reserved for FeatureKey separation"
            )
        if "__" in v:
            raise ValueError(
                f"project name '{v}' cannot contain double underscores (__). "
                f"Double underscores are reserved for table name generation"
            )
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(f"project name '{v}' must contain only alphanumeric characters, underscores, and hyphens")
        return v

    @property
    def plugins(self) -> list[str]:
        """Returns all enabled plugin names from ext configuration."""
        return [name for name, plugin in self.ext.items() if plugin.enable]

    @classmethod
    def get_plugin(cls, name: str, plugin_cls: type[PluginConfigT]) -> PluginConfigT:
        """Get the plugin config from the global Metaxy config.

        Unlike `get()`, this method does not warn when the global config is not
        initialized. This is intentional because plugins may call this at import
        time to read their configuration, and returning default plugin config
        is always safe.
        """
        ext = cls.get(_allow_default_config=True).ext
        if name in ext:
            existing = ext[name]
            if isinstance(existing, plugin_cls):
                # Already the correct type
                plugin = existing
            else:
                # Convert from generic PluginConfig or dict to specific plugin class
                plugin = plugin_cls.model_validate(existing.model_dump())
        else:
            # Return default config if plugin not configured
            plugin = plugin_cls()
        return plugin

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources: init → env → TOML.

        Priority (first wins):
        1. Init arguments
        2. Environment variables
        3. TOML file (with ``extend`` inheritance fully resolved)
        """
        # load() sets _toml_file_override before calling cls().
        # Bare MetaxyConfig() has no override → no TOML file.
        try:
            config_file = _toml_file_override.get()
        except LookupError:
            config_file = None
        toml_settings = MetaxyTomlSource(settings_cls, config_file=config_file)
        return (init_settings, env_settings, toml_settings)

    @classmethod
    def get(cls, *, load: bool = False, _allow_default_config: bool = False) -> "MetaxyConfig":
        """Get the current Metaxy configuration.

        Args:
            load: If True and config is not set, calls `MetaxyConfig.load()` to
                load configuration from file. Useful for plugins that need config
                but don't want to require manual initialization.
            _allow_default_config: Internal parameter. When True, returns default
                config without warning if global config is not set. Used by methods
                like `get_plugin` that may be called at import time.
        """
        cfg = _config_override.get() or _global_config
        if cfg is None:
            if load:
                return cls.load()
            if not _allow_default_config:
                warnings.warn(
                    UserWarning(
                        "Global Metaxy configuration not initialized. It can be set with MetaxyConfig.set(config) typically after loading it from a toml file. Returning default configuration (with environment variables and other pydantic settings sources resolved)."
                    ),
                    stacklevel=2,
                )
            return cls()
        else:
            return cfg

    @classmethod
    def set(cls, config: Self | None) -> None:
        """Set the current Metaxy configuration (visible to all threads)."""
        global _global_config
        _global_config = config

    @classmethod
    def is_set(cls) -> bool:
        """Check if the current Metaxy configuration is set."""
        return _config_override.get() is not None or _global_config is not None

    @classmethod
    def reset(cls) -> None:
        """Reset the current Metaxy configuration to None."""
        global _global_config
        _global_config = None

    @contextmanager
    def use(self) -> Iterator[Self]:
        """Use this configuration temporarily, restoring previous config on exit.

        Example:
            ```py
            test_config = MetaxyConfig(project="test")
            with test_config.use():
                # Code here uses test config
                assert MetaxyConfig.get().project == "test"
            # Previous config restored
            ```
        """
        token = _config_override.set(self)
        try:
            yield self
        finally:
            _config_override.reset(token)

    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        *,
        search_parents: bool = True,
        auto_discovery_start: Path | None = None,
    ) -> "MetaxyConfig":
        """Load config with auto-discovery and parent directory search.

        Args:
            config_file: Optional config file path.

                !!! tip
                    `METAXY_CONFIG` environment variable can be used to set this parameter

            search_parents: Search parent directories for config file
            auto_discovery_start: Directory to start search from.
                Defaults to current working directory.

        Returns:
            Loaded config (TOML + env vars merged)

        Example:
            <!-- skip next -->
            ```py
            # Auto-discover with parent search
            config = MetaxyConfig.load()

            # Explicit file
            config = MetaxyConfig.load("custom.toml")

            # Auto-discover without parent search
            config = MetaxyConfig.load(search_parents=False)

            # Auto-discover from a specific directory
            config = MetaxyConfig.load(auto_discovery_start=Path("/path/to/project"))
            ```
        """
        if config_from_env := os.getenv("METAXY_CONFIG"):
            config_file = Path(config_from_env)

        if config_file is None and search_parents:
            config_file = discover_config_with_parents(auto_discovery_start)

        resolved: Path | None = Path(config_file) if config_file else None

        # Pass config file path to settings_customise_sources via ContextVar
        token = _toml_file_override.set(resolved)
        try:
            config = cls()
        finally:
            _toml_file_override.reset(token)

        config._config_file = resolved.resolve() if resolved else None

        cls.set(config)

        # Load plugins after config is set (plugins may access MetaxyConfig.get())
        config._load_plugins()

        return config

    @overload
    def get_store(
        self,
        name: str | None = None,
        *,
        expected_type: Literal[None] = None,
        **kwargs: Any,
    ) -> "MetadataStore": ...

    @overload
    def get_store(
        self,
        name: str | None = None,
        *,
        expected_type: type[StoreTypeT],
        **kwargs: Any,
    ) -> StoreTypeT: ...

    def get_store(
        self,
        name: str | None = None,
        *,
        expected_type: type[StoreTypeT] | None = None,
        **kwargs: Any,
    ) -> "MetadataStore | StoreTypeT":
        """Instantiate metadata store by name.

        Args:
            name: Store name (uses config.store if None)
            expected_type: Expected type of the store.
                If the actual store type does not match the expected type, a `TypeError` is raised.
            **kwargs: Additional keyword arguments to pass to the store constructor.

        Returns:
            Instantiated metadata store

        Raises:
            ValueError: If store name not found in config, or if fallback stores
                have different hash algorithms than the parent store
            ImportError: If store class cannot be imported
            TypeError: If the actual store type does not match the expected type

        Example:
            ```py
            store = config.get_store("prod")

            # Use default store
            store = config.get_store()
            ```
        """
        from metaxy.versioning.types import HashAlgorithm

        if len(self.stores) == 0:
            raise InvalidConfigError.from_config(
                self,
                "No Metaxy stores available. They should be configured in metaxy.toml|pyproject.toml or via environment variables.",
            )

        name = name or self.store

        if name not in self.stores:
            raise InvalidConfigError.from_config(
                self,
                f"Store '{name}' not found in config. Available stores: {list(self.stores.keys())}",
            )

        store_config = self.stores[name]

        # Get store class (lazily imported on first access)
        try:
            store_class = store_config.type_cls
        except Exception as e:
            raise InvalidConfigError.from_config(
                self,
                f"Failed to import store class '{store_config.type}' for store '{name}': {e}",
            ) from e

        if expected_type is not None and not issubclass(store_class, expected_type):
            raise InvalidConfigError.from_config(
                self,
                f"Store '{name}' is not of type '{expected_type.__name__}'",
            )

        # Extract configuration and prepare for typed config model
        config_copy = store_config.config.copy()

        # Get hash_algorithm from config (if specified) and convert to enum
        configured_hash_algorithm = config_copy.get("hash_algorithm")
        if configured_hash_algorithm is not None:
            # Convert string to enum if needed
            if isinstance(configured_hash_algorithm, str):
                configured_hash_algorithm = HashAlgorithm(configured_hash_algorithm)
                config_copy["hash_algorithm"] = configured_hash_algorithm
        else:
            # Don't set a default here - let the store choose its own default
            configured_hash_algorithm = None

        # Get the store's config model class and create typed config
        config_model_cls = store_class.config_model()

        # Get auto_create_tables from global config only if the config model supports it
        if (
            "auto_create_tables" not in config_copy
            and self.auto_create_tables is not None
            and "auto_create_tables" in config_model_cls.model_fields
        ):
            # Use global setting from MetaxyConfig if not specified per-store
            config_copy["auto_create_tables"] = self.auto_create_tables

        # Separate kwargs into config fields and extra constructor args
        config_fields = set(config_model_cls.model_fields.keys())
        extra_kwargs = {}
        for key, value in kwargs.items():
            if key in config_fields:
                config_copy[key] = value
            else:
                extra_kwargs[key] = value

        try:
            typed_config = config_model_cls.model_validate(config_copy)
        except Exception as e:
            raise InvalidConfigError.from_config(
                self,
                f"Failed to validate config for store '{name}': {e}",
            ) from e

        # Instantiate using from_config() - fallback stores are resolved via MetaxyConfig.get()
        # Use self.use() to ensure this config is available for fallback resolution
        try:
            with self.use():
                store = store_class.from_config(typed_config, name=name, **extra_kwargs)
        except InvalidConfigError:
            # Don't re-wrap InvalidConfigError (e.g., from nested fallback store resolution)
            raise
        except Exception as e:
            raise InvalidConfigError.from_config(
                self,
                f"Failed to instantiate store '{name}' ({store_class.__name__}): {e}",
            ) from e

        # Verify the store actually uses the hash algorithm we configured
        # (in case a store subclass overrides the default or ignores the parameter)
        # Only check if we explicitly configured a hash algorithm
        if configured_hash_algorithm is not None and store.hash_algorithm != configured_hash_algorithm:
            raise InvalidConfigError.from_config(
                self,
                f"Store '{name}' ({store_class.__name__}) was configured with "
                f"hash_algorithm='{configured_hash_algorithm.value}' but is using "
                f"'{store.hash_algorithm.value}'. The store class may have overridden "
                f"the hash algorithm. All stores must use the same hash algorithm.",
            )

        if expected_type is not None and not isinstance(store, expected_type):
            raise InvalidConfigError.from_config(
                self,
                f"Store '{name}' is not of type '{expected_type.__name__}'",
            )

        return store

    def to_toml(self) -> str:
        """Serialize to TOML string.

        Returns:
            TOML representation of this configuration.
        """
        data = self.model_dump(mode="json", by_alias=True)
        # Remove None values (TOML doesn't support them)
        data = _remove_none_values(data)
        return tomli_w.dumps(data)
