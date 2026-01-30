"""Configuration system for Metaxy using pydantic-settings."""
# pyright: reportImportCycles=false

import os
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import tomli
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

if TYPE_CHECKING:
    from metaxy.metadata_store.base import (
        MetadataStore,
    )

T = TypeVar("T")

# Pattern for ${VAR} or ${VAR:-default} syntax
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _collect_dict_keys(d: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively collect all keys from a nested dict as dot-separated paths."""
    keys = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        keys.append(full_key)
        if isinstance(value, dict):
            keys.extend(_collect_dict_keys(value, full_key))
    return keys


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


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values.

    Supports:
    - ${VAR} - substitutes with environment variable value, empty string if not set
    - ${VAR:-default} - substitutes with environment variable value, or default if not set

    Args:
        value: The value to expand (can be string, dict, list, or other)

    Returns:
        The value with environment variables expanded
    """
    if isinstance(value, str):

        def replace_match(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)  # None if no default specified
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                return ""

        return _ENV_VAR_PATTERN.sub(replace_match, value)
    elif isinstance(value, Mapping):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, Sequence):
        return [_expand_env_vars(item) for item in value]
    else:
        return value


class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for TOML configuration files.

    Auto-discovers configuration in this order:
    1. Explicit file path if provided
    2. metaxy.toml in current directory (preferred)
    3. pyproject.toml [tool.metaxy] section (fallback)
    4. No config (returns empty dict)
    """

    def __init__(self, settings_cls: type[BaseSettings], toml_file: Path | None = None):
        super().__init__(settings_cls)
        self.toml_file = toml_file or self._discover_config_file()
        self.toml_data = self._load_toml()

    def _discover_config_file(self) -> Path | None:
        """Auto-discover config file."""
        # Prefer metaxy.toml
        if Path("metaxy.toml").exists():
            return Path("metaxy.toml")

        # Fallback to pyproject.toml
        if Path("pyproject.toml").exists():
            return Path("pyproject.toml")

        return None

    def _load_toml(self) -> dict[str, Any]:
        """Load TOML file and extract metaxy config.

        Environment variables in the format ${VAR} or ${VAR:-default} are
        expanded in string values.
        """
        if self.toml_file is None:
            return {}

        with open(self.toml_file, "rb") as f:
            data = tomli.load(f)

        # Extract [tool.metaxy] from pyproject.toml or root from metaxy.toml
        if self.toml_file.name == "pyproject.toml":
            config = data.get("tool", {}).get("metaxy", {})
        else:
            config = data

        # Expand environment variables in config values
        return _expand_env_vars(config)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Get field value from TOML data."""
        field_value = self.toml_data.get(field_name)
        return field_value, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return all settings from TOML."""
        return self.toml_data


@public
class StoreConfig(BaseSettings):
    """Configuration for a single metadata store.

    Example:
        ```py
        store_config = StoreConfig(
            type="metaxy_delta.DeltaMetadataStore",
            config={
                "root_path": "s3://bucket/metadata",
                "region": "us-west-2",
                "fallback_stores": ["prod"],
            },
        )
        ```
    """

    model_config = SettingsConfigDict(
        extra="forbid",  # Only type and config fields allowed
        frozen=True,
        populate_by_name=True,  # Allow both 'type' and 'type_path' in constructor
    )

    # Store the import path as string (internal field)
    # Uses alias="type" so TOML and constructor use "type"
    # Annotated as str | type to allow passing class objects directly
    type_path: str | type[Any] = PydanticField(
        alias="type",
        description="Full import path to metadata store class (e.g., 'metaxy.metadata_store.duckdb.DuckDBMetadataStore')",
    )

    config: dict[str, Any] = PydanticField(
        default_factory=dict,
        description="Store-specific configuration parameters (kwargs for __init__). Includes fallback_stores, database paths, connection parameters, etc.",
    )

    @field_validator("type_path", mode="before")
    @classmethod
    def _coerce_type_to_string(cls, v: Any) -> str:
        """Accept both string import paths and class objects.

        Converts class objects to their full import path string.
        """
        if isinstance(v, str):
            return v
        if isinstance(v, type):
            # Convert class to import path string
            return f"{v.__module__}.{v.__qualname__}"
        raise ValueError(f"type must be a string or class, got {type(v).__name__}")

    @cached_property
    def type(self) -> type[Any]:
        """Get the store class, importing lazily on first access.

        Returns:
            The metadata store class

        Raises:
            ImportError: If the store class cannot be imported
        """
        import importlib

        from pydantic import TypeAdapter
        from pydantic.types import ImportString

        adapter: TypeAdapter[type[Any]] = TypeAdapter(ImportString[Any])
        try:
            return adapter.validate_python(self.type_path)
        except Exception:
            # Pydantic's ImportString swallows the underlying ImportError for other packages/modules,
            # showing a potentially misleading message.
            # Try a direct import to surface the real error (e.g., missing dependency).
            module_path, _, _ = str(self.type_path).rpartition(".")
            if module_path:
                try:
                    importlib.import_module(module_path)
                except ImportError as import_err:
                    raise ImportError(f"Cannot import '{self.type_path}': {import_err}") from import_err
            raise

    def to_toml(self) -> str:
        """Serialize to TOML string.

        Returns:
            TOML representation of this store configuration.
        """
        data = self.model_dump(mode="json", by_alias=True)
        return tomli_w.dumps(data)


class PluginConfig(BaseSettings):
    """Configuration for Metaxy plugins"""

    model_config = SettingsConfigDict(frozen=True, extra="allow")

    enable: bool = PydanticField(
        default=False,
        description="Whether to enable the plugin.",
    )


PluginConfigT = TypeVar("PluginConfigT", bound=PluginConfig)

# Context variable for storing the app context
_metaxy_config: ContextVar["MetaxyConfig | None"] = ContextVar("_metaxy_config", default=None)


BUILTIN_PLUGINS = {
    "sqlmodel": "metaxy.ext.sqlmodel",
    "alembic": "metaxy.ext.alembic",
}


StoreTypeT = TypeVar("StoreTypeT", bound="MetadataStore")


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

    Example: Templating environment variables
        ```toml {title="metaxy.toml"}
        [stores.branch.config]
        root_path = "s3://my-bucket/${BRANCH_NAME}"
        ```

    The default store is `"dev"`; `METAXY_STORE` can be used to override it.

    Incomplete store configurations are filtered out if the store type is not set.
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
            is_complete = isinstance(config, StoreConfig) or (
                isinstance(config, dict) and ("type" in config or "type_path" in config)
            )
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

    migrations_dir: str = PydanticField(
        default=".metaxy/migrations",
        description="Directory where migration files are stored",
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

    hash_truncation_length: int | None = PydanticField(
        default=None,
        description="Truncate hash values to this length (minimum 8 characters).",
    )

    auto_create_tables: bool = PydanticField(
        default=False,
        description="Auto-create tables when opening stores (development/testing only). WARNING: Do not use in production. Use proper database migration tools like Alembic.",
    )

    project: str | None = PydanticField(
        default=None,
        description="Project name for metadata isolation. Used to scope operations to enable multiple independent projects in a shared metadata store. Does not modify feature keys or table names. Project names must be valid alphanumeric strings with dashes, underscores, and cannot contain forward slashes (`/`) or double underscores (`__`)",
    )

    locked: bool | None = PydanticField(
        default=None,
        description="Whether to raise an error if an external feature doesn't have a matching feature version when [syncing external features][metaxy.sync_external_features] from the metadata store.",
    )

    # Private attribute to track which config file was used (set by load())
    _config_file: Path | None = PrivateAttr(default=None)

    @property
    def config_file(self) -> Path | None:
        """The config file path used to load this configuration.

        Returns None if the config was created directly (not via load()).
        """
        return self._config_file

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

    @field_validator("hash_truncation_length")
    @classmethod
    def validate_hash_truncation_length(cls, v: int | None) -> int | None:
        """Validate hash truncation length is at least 8 if set."""
        if v is not None and v < 8:
            raise ValueError(f"hash_truncation_length must be at least 8 characters, got {v}")
        return v

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
        3. TOML file
        """
        toml_settings = TomlConfigSettingsSource(settings_cls)
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
        cfg = _metaxy_config.get()
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
        """Set the current Metaxy configuration."""
        _metaxy_config.set(config)

    @classmethod
    def is_set(cls) -> bool:
        """Check if the current Metaxy configuration is set."""
        return _metaxy_config.get() is not None

    @classmethod
    def reset(cls) -> None:
        """Reset the current Metaxy configuration to None."""
        _metaxy_config.set(None)

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
        previous = _metaxy_config.get()
        _metaxy_config.set(self)
        try:
            yield self
        finally:
            _metaxy_config.set(previous)

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
        # Search for config file if not explicitly provided

        if config_from_env := os.getenv("METAXY_CONFIG"):
            config_file = Path(config_from_env)

        if config_file is None and search_parents:
            config_file = cls._discover_config_with_parents(auto_discovery_start)

        # For explicit file, temporarily patch the TomlConfigSettingsSource
        # to use that file, then use normal instantiation
        # This ensures env vars still work

        if config_file:
            # Create a custom settings source class for this file
            toml_path = Path(config_file)

            class CustomTomlSource(TomlConfigSettingsSource):
                def __init__(self, settings_cls: type[BaseSettings]):
                    # Skip auto-discovery, use explicit file
                    super(TomlConfigSettingsSource, self).__init__(settings_cls)
                    self.toml_file = toml_path
                    self.toml_data = self._load_toml()

            # Customize sources to use custom TOML file
            original_method = cls.settings_customise_sources

            @classmethod
            def custom_sources(
                cls_inner,
                settings_cls,
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
            ):
                toml_settings = CustomTomlSource(settings_cls)
                return (init_settings, env_settings, toml_settings)

            # Temporarily replace method
            cls.settings_customise_sources = custom_sources  # ty: ignore[invalid-assignment]
            config = cls()
            cls.settings_customise_sources = original_method  # ty: ignore[invalid-assignment]
            # Store the resolved config file path
            config._config_file = toml_path.resolve()
        else:
            # Use default sources (auto-discovery + env vars)
            config = cls()
            # No config file used
            config._config_file = None

        cls.set(config)

        # Load plugins after config is set (plugins may access MetaxyConfig.get())
        config._load_plugins()

        return config

    @staticmethod
    def _discover_config_with_parents(start_dir: Path | None = None) -> Path | None:
        """Discover config file by searching current and parent directories.

        Searches for metaxy.toml or pyproject.toml in start directory,
        then iteratively searches parent directories.

        Args:
            start_dir: Directory to start search from (defaults to cwd)

        Returns:
            Path to config file if found, None otherwise
        """
        current = start_dir or Path.cwd()

        while True:
            # Check for metaxy.toml (preferred)
            metaxy_toml = current / "metaxy.toml"
            if metaxy_toml.exists():
                return metaxy_toml

            # Check for pyproject.toml
            pyproject_toml = current / "pyproject.toml"
            if pyproject_toml.exists():
                return pyproject_toml

            # Move to parent
            parent = current.parent
            if parent == current:
                # Reached roothash_tru
                break
            current = parent

        return None

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
            store_class = store_config.type
        except Exception as e:
            raise InvalidConfigError.from_config(
                self,
                f"Failed to import store class '{store_config.type_path}' for store '{name}': {e}",
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
                store = store_class.from_config(typed_config, **extra_kwargs)
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


def _remove_none_values(obj: Any) -> Any:
    """Recursively remove None values from a dict (TOML doesn't support None)."""
    if isinstance(obj, dict):
        return {k: _remove_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_remove_none_values(item) for item in obj if item is not None]
    return obj
