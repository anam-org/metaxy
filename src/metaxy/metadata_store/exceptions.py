"""Exceptions for metadata store operations."""

from metaxy._decorators import public


@public
class MetadataStoreError(Exception):
    """Base exception for metadata store errors."""

    pass


@public
class FeatureNotFoundError(MetadataStoreError):
    """Raised when a feature is not found in the store."""

    pass


@public
class SystemDataNotFoundError(MetadataStoreError):
    """Raised when system features are not found in the store."""

    pass


@public
class FieldNotFoundError(MetadataStoreError):
    """Raised when a field is not found for a feature."""

    pass


@public
class MetadataSchemaError(MetadataStoreError):
    """Raised when metadata DataFrame has invalid schema."""

    pass


@public
class DependencyError(MetadataStoreError):
    """Raised when upstream dependencies are missing or invalid."""

    pass


@public
class StoreNotOpenError(MetadataStoreError):
    """Raised when attempting to use a store that hasn't been opened."""

    pass


@public
class HashAlgorithmNotSupportedError(MetadataStoreError):
    """Raised when a hash algorithm is not supported by the store or its components."""

    pass


@public
class TableNotFoundError(MetadataStoreError):
    """Raised when a table does not exist and auto_create_tables is disabled."""

    pass


@public
class VersioningEngineMismatchError(Exception):
    """Raised when versioning_engine='native' is requested but data has wrong implementation."""

    pass
