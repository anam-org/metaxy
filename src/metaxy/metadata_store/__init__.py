"""Metadata store for feature pipeline management."""

from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.exceptions import (
    ContainerNotFoundError,
    DependencyError,
    FeatureNotFoundError,
    HashAlgorithmNotSupportedError,
    MetadataSchemaError,
    MetadataStoreError,
    StoreNotOpenError,
)
from metaxy.metadata_store.memory import InMemoryMetadataStore

__all__ = [
    "MetadataStore",
    "InMemoryMetadataStore",
    "MetadataStoreError",
    "FeatureNotFoundError",
    "ContainerNotFoundError",
    "MetadataSchemaError",
    "DependencyError",
    "StoreNotOpenError",
    "HashAlgorithmNotSupportedError",
    "FEATURE_VERSIONS_KEY",
    "MIGRATION_HISTORY_KEY",
    "allow_feature_version_override",
]
