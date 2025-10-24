"""Metadata store for feature pipeline management."""

from metaxy.metadata_store.base import (
    FEATURE_VERSIONS_KEY,
    MIGRATION_HISTORY_KEY,
    MetadataStore,
    allow_feature_version_override,
)
from metaxy.metadata_store.ducklake import DuckLakeMetadataStore
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

__all__ = [
    "MetadataStore",
    "InMemoryMetadataStore",
    "DuckLakeMetadataStore",
    "MetadataStoreError",
    "FeatureNotFoundError",
    "FieldNotFoundError",
    "MetadataSchemaError",
    "DependencyError",
    "StoreNotOpenError",
    "HashAlgorithmNotSupportedError",
    "FEATURE_VERSIONS_KEY",
    "MIGRATION_HISTORY_KEY",
    "allow_feature_version_override",
]
