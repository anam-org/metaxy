"""Exceptions for metadata store operations."""


class MetadataStoreError(Exception):
    """Base exception for metadata store errors."""

    pass


class FeatureNotFoundError(MetadataStoreError):
    """Raised when a feature is not found in the store."""

    pass


class SystemDataNotFoundError(MetadataStoreError):
    """Raised when system features are not found in the store."""

    pass


class FieldNotFoundError(MetadataStoreError):
    """Raised when a field is not found for a feature."""

    pass


class MetadataSchemaError(MetadataStoreError):
    """Raised when metadata DataFrame has invalid schema."""

    pass


class DependencyError(MetadataStoreError):
    """Raised when upstream dependencies are missing or invalid."""

    pass


class StoreNotOpenError(MetadataStoreError):
    """Raised when attempting to use a store that hasn't been opened."""

    pass


class HashAlgorithmNotSupportedError(MetadataStoreError):
    """Raised when a hash algorithm is not supported by the store or its components."""

    pass


class TableNotFoundError(MetadataStoreError):
    """Raised when a table does not exist and auto_create_tables is disabled."""

    pass


class VersioningEngineMismatchError(Exception):
    """Raised when versioning_engine='native' is requested but data has wrong implementation."""

    pass
