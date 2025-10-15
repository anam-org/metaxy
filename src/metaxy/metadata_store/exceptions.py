"""Exceptions for metadata store operations."""


class MetadataStoreError(Exception):
    """Base exception for metadata store errors."""

    pass


class FeatureNotFoundError(MetadataStoreError):
    """Raised when a feature is not found in the store."""

    pass


class ContainerNotFoundError(MetadataStoreError):
    """Raised when a container is not found for a feature."""

    pass


class MetadataSchemaError(MetadataStoreError):
    """Raised when metadata DataFrame has invalid schema."""

    pass


class DependencyError(MetadataStoreError):
    """Raised when upstream dependencies are missing or invalid."""

    pass


class StoreNotOpenError(MetadataStoreError):
    """Raised when attempting to use a store outside of a context manager."""

    pass
