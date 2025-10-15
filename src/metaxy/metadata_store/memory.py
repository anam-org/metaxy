"""In-memory metadata store implementation."""

import polars as pl

from metaxy.metadata_store.base import MetadataStore
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
        # Use tuple as key (hashable) instead of string to avoid parsing issues
        self._storage: dict[tuple[str, ...], pl.DataFrame] = {}

    def open(self) -> None:
        """Open the store (no-op for in-memory store)."""
        pass

    def close(self) -> None:
        """Close the store (no-op for in-memory store)."""
        pass

    def _get_storage_key(self, feature_key: FeatureKey) -> tuple[str, ...]:
        """Convert feature key to storage key (tuple for hashability)."""
        return tuple(feature_key)

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """
        Internal write implementation for in-memory storage.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated)
        """
        storage_key = self._get_storage_key(feature_key)

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
        feature_key = self._resolve_feature_key(feature)
        storage_key = self._get_storage_key(feature_key)

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
            List of FeatureKey objects (excluding system tables)
        """
        features = []
        for key_tuple in self._storage.keys():
            # Convert tuple back to FeatureKey
            feature_key = FeatureKey(list(key_tuple))

            # Skip system tables
            if not self._is_system_table(feature_key):
                features.append(feature_key)

        return features

    def clear(self) -> None:
        """
        Clear all metadata from store.

        Useful for testing.
        """
        self._storage.clear()

    def display(self) -> str:
        """Display string for this store."""
        num_features = len(self._storage)
        return f"InMemoryMetadataStore(features={num_features})"
