"""In-memory metadata store implementation."""

import polars as pl

from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.exceptions import MetadataSchemaError
from metaxy.models.feature import Feature
from metaxy.models.types import FeatureKey


class InMemoryMetadataStore(MetadataStore):
    """
    In-memory metadata store using dict-based storage.

    Features:
    - Simple dict storage: {FeatureKey: pl.DataFrame}
    - Fast for testing and prototyping
    - No persistence (data lost when process exits)
    - Schema validation on write
    - Polars-only computation (no native optimization)

    Limitations:
    - Not suitable for production
    - Data lost on process exit
    - No concurrency support across processes
    - Memory-bound (all data in RAM)
    """

    def __init__(self, **kwargs):
        """
        Initialize in-memory store.

        Args:
            **kwargs: Passed to MetadataStore.__init__ (e.g., fallback_stores)
        """
        super().__init__(**kwargs)
        self._storage: dict[str, pl.DataFrame] = {}

    def _get_storage_key(self, feature: FeatureKey | type[Feature]) -> str:
        """Convert feature to storage key (string)."""
        feature_key = self._resolve_feature_key(feature)
        return feature_key.to_string()

    def _validate_schema(self, df: pl.DataFrame) -> None:
        """
        Validate that DataFrame has required schema.

        Args:
            df: DataFrame to validate

        Raises:
            MetadataSchemaError: If schema is invalid
        """
        # Check for data_version column
        if "data_version" not in df.columns:
            raise MetadataSchemaError("DataFrame must have 'data_version' column")

        # Check that data_version is a struct
        data_version_type = df.schema["data_version"]
        if not isinstance(data_version_type, pl.Struct):
            raise MetadataSchemaError(
                f"'data_version' column must be pl.Struct, got {data_version_type}"
            )

    def write_metadata(
        self,
        feature: FeatureKey | type[Feature],
        df: pl.DataFrame,
    ) -> None:
        """
        Write metadata for a feature (immutable, append-only).

        Args:
            feature: Feature to write metadata for
            df: DataFrame with metadata (must have 'data_version' struct column)

        Raises:
            MetadataSchemaError: If DataFrame schema is invalid
        """
        # Validate schema
        self._validate_schema(df)

        storage_key = self._get_storage_key(feature)

        # Append or create
        if storage_key in self._storage:
            # Append to existing
            self._storage[storage_key] = pl.concat(
                [self._storage[storage_key], df],
                how="vertical",
            )
        else:
            # Create new
            self._storage[storage_key] = df

    def _read_metadata_local(
        self,
        feature: FeatureKey | type[Feature],
        *,
        filters: pl.Expr | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame | None:
        """
        Read metadata from this store only (no fallback).

        Args:
            feature: Feature to read
            filters: Optional Polars filter expression
            columns: Optional list of columns to select

        Returns:
            DataFrame with metadata, or None if not found
        """
        storage_key = self._get_storage_key(feature)

        if storage_key not in self._storage:
            return None

        df = self._storage[storage_key]

        # Apply filters
        if filters is not None:
            df = df.filter(filters)

        # Select columns
        if columns is not None:
            df = df.select(columns)

        return df

    def _list_features_local(self) -> list[FeatureKey]:
        """
        List all features in this store.

        Returns:
            List of FeatureKey objects
        """
        features = []
        for key_str in self._storage.keys():
            # Convert string back to FeatureKey
            # Assume keys are stored as underscore-separated strings
            parts = key_str.split("_")
            features.append(FeatureKey(parts))

        return features

    def clear(self) -> None:
        """
        Clear all metadata from store.

        Useful for testing.
        """
        self._storage.clear()

    def __repr__(self) -> str:
        """String representation."""
        num_features = len(self._storage)
        num_fallbacks = len(self.fallback_stores)
        return (
            f"InMemoryMetadataStore("
            f"features={num_features}, "
            f"fallback_stores={num_fallbacks})"
        )
