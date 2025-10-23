"""Configuration system for Metaxy using pydantic-settings."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, overload

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python 3.10

from pydantic import Field as PydanticField
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import FeatureRegistry

T = TypeVar("T")


def load_registry(registry_path: str) -> "FeatureRegistry":
    """Load registry from import path.

    Args:
        registry_path: Import path like "myapp.features:my_registry"

    Returns:
        FeatureRegistry instance

    Raises:
        ValueError: If path is invalid
        ImportError: If module or registry not found
        TypeError: If loaded object is not a FeatureRegistry
    """
    from metaxy.models.feature import FeatureRegistry

    if ":" not in registry_path:
        raise ValueError(
            f"Invalid registry path: {registry_path}. "
            f"Expected format: 'module.path:registry_name'"
        )

    module_path, registry_name = registry_path.split(":", 1)

    try:
        module = __import__(module_path, fromlist=[registry_name])
        registry = getattr(module, registry_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load registry from {registry_path}: {e}") from e

    if not isinstance(registry, FeatureRegistry):
        raise TypeError(
            f"Registry at {registry_path} is not a FeatureRegistry instance. "
            f"Got: {type(registry)}"
        )

    return registry


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
        """Load TOML file and extract metaxy config."""
        if self.toml_file is None:
            return {}

        with open(self.toml_file, "rb") as f:
            data = tomllib.load(f)

        # Extract [tool.metaxy] from pyproject.toml or root from metaxy.toml
        if self.toml_file.name == "pyproject.toml":
            return data.get("tool", {}).get("metaxy", {})
        else:
            return data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Get field value from TOML data."""
        field_value = self.toml_data.get(field_name)
        return field_value, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return all settings from TOML."""
        return self.toml_data


class StoreConfig(BaseSettings):
    """Configuration for a single metadata store.

    Structure:
        type: Full import path to store class
        config: Dict of all configuration (including fallback_stores)

    Example:
        >>> config = StoreConfig(
        ...     type="metaxy_delta.DeltaMetadataStore",
        ...     config={
        ...         "table_uri": "s3://bucket/metadata",
        ...         "region": "us-west-2",
        ...         "fallback_stores": ["prod"],
        ...     }
        ... )
    """

    model_config = SettingsConfigDict(
        extra="forbid",  # Only type and config fields allowed
    )

    # Store class (full import path)
    type: str

    # Store configuration (all kwargs for __init__)
    # This includes fallback_stores, table_uri, db_path, storage_options, etc.
    config: dict[str, Any] = PydanticField(default_factory=dict)


class MetaxyConfig(BaseSettings):
    """Main Metaxy configuration.

    Loads from:
    1. TOML file (metaxy.toml or pyproject.toml [tool.metaxy])
    2. Environment variables (METAXY_*)
    3. Init arguments

    Priority: init > env vars > TOML

    Example:
        >>> # Auto-discover config
        >>> config = MetaxyConfig.load()
        >>>
        >>> # Get store instance
        >>> store = config.get_store("prod")
        >>>
        >>> # Override via env var
        >>> # METAXY_STORE=staging METAXY_REGISTRY=myapp.features:my_registry
        >>> config = MetaxyConfig.load()
        >>> store = config.get_store()  # Uses staging with custom registry
    """

    model_config = SettingsConfigDict(
        env_prefix="METAXY_",
        env_nested_delimiter="__",
    )

    # Store to use
    store: str = "dev"

    # Named store configurations
    stores: dict[str, StoreConfig] = PydanticField(default_factory=dict)

    # Migrations directory
    migrations_dir: str = "metaxy/migrations"

    # Entrypoints to load (list of module paths)
    entrypoints: list[str] = PydanticField(default_factory=list)

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
    def load(
        cls, config_file: str | Path | None = None, *, search_parents: bool = True
    ) -> "MetaxyConfig":
        """Load config with auto-discovery and parent directory search.

        Args:
            config_file: Optional config file path (overrides auto-discovery)
            search_parents: Search parent directories for config file (default: True)

        Returns:
            Loaded config (TOML + env vars merged)

        Example:
            >>> # Auto-discover with parent search
            >>> config = MetaxyConfig.load()
            >>>
            >>> # Explicit file
            >>> config = MetaxyConfig.load("custom.toml")

            >>> # Auto-discover without parent search
            >>> config = MetaxyConfig.load(search_parents=False)
        """
        # Search for config file if not explicitly provided
        if config_file is None and search_parents:
            config_file = cls._discover_config_with_parents()

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

            @classmethod  # type: ignore[misc]
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
            cls.settings_customise_sources = custom_sources  # type: ignore[assignment]
            config = cls()
            cls.settings_customise_sources = original_method  # type: ignore[method-assign]
        else:
            # Use default sources (auto-discovery + env vars)
            config = cls()

        # Load entrypoints if configured (do this after config is created)
        if config.entrypoints:
            from metaxy.entrypoints import load_config_entrypoints

            load_config_entrypoints(config.entrypoints)

        return config

    @staticmethod
    def _discover_config_with_parents() -> Path | None:
        """Discover config file by searching current and parent directories.

        Searches for metaxy.toml or pyproject.toml in current directory,
        then iteratively searches parent directories.

        Returns:
            Path to config file if found, None otherwise
        """
        current = Path.cwd()

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
                # Reached root
                break
            current = parent

        return None

    def get_store(
        self,
        name: str | None = None,
    ) -> "MetadataStore":
        """Instantiate metadata store by name.

        Args:
            name: Store name (uses config.store if None)

        Returns:
            Instantiated metadata store

        Raises:
            ValueError: If store name not found in config, or if fallback stores
                have different hash algorithms than the parent store
            ImportError: If store class cannot be imported

        Example:
            >>> config = MetaxyConfig.load()
            >>> store = config.get_store("prod")
            >>>
            >>> # Use default store
            >>> store = config.get_store()
        """
        from metaxy.data_versioning.hash_algorithms import HashAlgorithm

        if len(self.stores) == 0:
            raise ValueError(
                "No Metaxy stores available. They should be configured in metaxy.toml|pyproject.toml or via environment variables."
            )

        name = name or self.store

        if name not in self.stores:
            raise ValueError(
                f"Store '{name}' not found in config. "
                f"Available stores: {list(self.stores.keys())}"
            )

        store_config = self.stores[name]

        # Import store class
        store_class = self._import_class(store_config.type)

        # Extract configuration
        config_copy = store_config.config.copy()
        fallback_store_names = config_copy.pop("fallback_stores", [])

        # Get hash_algorithm from config (if specified) and convert to enum
        configured_hash_algorithm = config_copy.get("hash_algorithm")
        if configured_hash_algorithm is not None:
            # Convert string to enum if needed
            if isinstance(configured_hash_algorithm, str):
                configured_hash_algorithm = HashAlgorithm(configured_hash_algorithm)
                config_copy["hash_algorithm"] = configured_hash_algorithm
        else:
            # Use default
            configured_hash_algorithm = HashAlgorithm.XXHASH64
            config_copy["hash_algorithm"] = configured_hash_algorithm

        # Build fallback stores recursively
        fallback_stores = []
        for fallback_name in fallback_store_names:
            fallback_store = self.get_store(fallback_name)
            fallback_stores.append(fallback_store)

        # Instantiate store with config + fallback_stores
        store = store_class(
            fallback_stores=fallback_stores,
            **config_copy,
        )

        # Verify the store actually uses the hash algorithm we configured
        # (in case a store subclass overrides the default or ignores the parameter)
        if store.hash_algorithm != configured_hash_algorithm:
            raise ValueError(
                f"Store '{name}' ({store_class.__name__}) was configured with "
                f"hash_algorithm='{configured_hash_algorithm.value}' but is using "
                f"'{store.hash_algorithm.value}'. The store class may have overridden "
                f"the hash algorithm. All stores must use the same hash algorithm."
            )

        return store

    @staticmethod
    def _import_class(class_path: str) -> type:
        """Import class from module path.

        Args:
            class_path: Full import path like "metaxy.metadata_store.InMemoryMetadataStore"

        Returns:
            Imported class

        Raises:
            ImportError: If module or class not found
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    @staticmethod
    @overload
    def _import_object(import_path: str, expected_type: type[T]) -> T: ...

    @staticmethod
    @overload
    def _import_object(import_path: str, expected_type: None = None) -> Any: ...

    @staticmethod
    def _import_object(
        import_path: str, expected_type: type[T] | None = None
    ) -> T | Any:
        """Import object from module:object path.

        Args:
            import_path: Import path like "module.path:object_name"
            expected_type: Optional type to validate against

        Returns:
            Imported object (typed as expected_type if provided)

        Raises:
            ValueError: If path format is invalid
            ImportError: If module or object not found
            TypeError: If object doesn't match expected_type
        """
        if ":" not in import_path:
            raise ValueError(
                f"Invalid import path: {import_path}. "
                f"Expected format: 'module.path:object_name'"
            )

        module_path, object_name = import_path.split(":", 1)

        try:
            module = __import__(module_path, fromlist=[object_name])
            obj = getattr(module, object_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load object from {import_path}: {e}") from e

        if expected_type is not None and not isinstance(obj, expected_type):
            type_name = getattr(expected_type, "__name__", str(expected_type))
            raise TypeError(
                f"Object at {import_path} is not a {type_name} instance. "
                f"Got: {type(obj)}"
            )

        return obj
