"""Metadata store for feature pipeline management."""

from metaxy.metadata_store.base import MetadataStore, allow_feature_version_override
from metaxy.metadata_store.exceptions import (
    DependencyError,
    FeatureNotFoundError,
    FieldNotFoundError,
    HashAlgorithmNotSupportedError,
    MetadataSchemaError,
    MetadataStoreError,
    StoreNotOpenError,
)
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.metadata_store.system import (
    FEATURE_VERSIONS_KEY,
)
from metaxy.metadata_store.types import AccessMode

if TYPE_CHECKING:
    from typing import Any

    from metaxy.metadata_store.lancedb import LanceDBMetadataStore


def _optional_import(name: str) -> Any | None:
    """Import optional metadata store modules on demand."""
    try:
        module = import_module(f"metaxy.metadata_store.{name}")
    except ModuleNotFoundError:
        return None
    return module


_delta_module = _optional_import("delta")
_lancedb_module = _optional_import("lancedb")


__all__ = [
    "MetadataStore",
    "InMemoryMetadataStore",
    "MetadataStoreError",
    "FeatureNotFoundError",
    "FieldNotFoundError",
    "MetadataSchemaError",
    "DependencyError",
    "StoreNotOpenError",
    "HashAlgorithmNotSupportedError",
    "FEATURE_VERSIONS_KEY",
    "allow_feature_version_override",
    "AccessMode",
    "LanceDBMetadataStore",
]
