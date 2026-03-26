from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeAlias

import tomli_w
from pydantic import Field as PydanticField
from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from metaxy._decorators import public

if TYPE_CHECKING:
    from metaxy.metadata_store.base import (
        MetadataStore,
    )

MetadataStoreType: TypeAlias = type["MetadataStore"]


@public
class StoreConfig(BaseSettings):
    """Configuration options for metadata stores."""

    model_config = SettingsConfigDict(
        extra="forbid",  # Only type and config fields allowed
        frozen=True,
    )

    # Store the import path as string (internal field)
    # Uses alias="type" so TOML and constructor use "type"
    # Annotated as str | type to allow passing class objects directly
    type: str = PydanticField(
        description='Full import path to metadata store class (e.g., `"metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"`)',
    )

    config: dict[str, Any] = PydanticField(
        default_factory=dict,
        description="Store-specific configuration parameters (constructor kwargs). Includes `fallback_stores`, database connection parameters, etc.",
    )

    @field_validator("type", mode="before")
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
    def type_cls(self) -> MetadataStoreType:
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
            return adapter.validate_python(self.type)
        except Exception:
            # Pydantic's ImportString swallows the underlying ImportError for other packages/modules,
            # showing a potentially misleading message.
            # Try a direct import to surface the real error (e.g., missing dependency).
            module_path, _, _ = str(self.type).rpartition(".")
            if module_path:
                try:
                    importlib.import_module(module_path)
                except ImportError as import_err:
                    raise ImportError(f"Cannot import '{self.type}': {import_err}") from import_err
            raise

    def to_toml(self) -> str:
        """Serialize to TOML string.

        Returns:
            TOML representation of this store configuration.
        """
        data = self.model_dump(mode="json", by_alias=True)
        return tomli_w.dumps(data)
