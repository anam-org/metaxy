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
from metaxy.metadata_store.system import (
    FEATURE_VERSIONS_KEY,
)
from metaxy.metadata_store.types import AccessMode

__all__ = [
    "MetadataStore",
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
]
