"""In-memory metadata store implementation."""

import polars as pl

from metaxy.data_versioning.calculators.base import DataVersionCalculator
from metaxy.data_versioning.diff.base import MetadataDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import Feature
from metaxy.models.types import FeatureKey


class InMemoryMetadataStore(MetadataStore[pl.LazyFrame]):
    """
    In-memory metadata store using dict-based storage.

    Features:
    - Simple dict storage: {FeatureKey: pl.DataFrame}
    - Fast for testing and prototyping
    - No persistence (data lost when process exits)
    - Schema validation on write
    - Uses Polars components for all operations

    Limitations:
    - Not suitable for production
    - Data lost on process exit
    - No concurrency support across processes
    - Memory-bound (all data in RAM)

    Type Parameters:
        TRef = pl.LazyFrame (uses Polars LazyFrames)

    Components:
        Components are created on-demand in resolve_update().
        Only supports Polars components (no native backend).
    """

    def __init__(self, **kwargs):
        """
        Initialize in-memory store.

        Args:
            **kwargs: Passed to MetadataStore.__init__ (e.g., fallback_stores, hash_algorithm)
        """
        # Use tuple as key (hashable) instead of string to avoid parsing issues
        self._storage: dict[tuple[str, ...], pl.DataFrame] = {}
        super().__init__(**kwargs)

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for in-memory store."""
        return HashAlgorithm.XXHASH64

    def _get_storage_key(self, feature_key: FeatureKey) -> tuple[str, ...]:
        """Convert feature key to storage key (tuple for hashability)."""
        return tuple(feature_key)

    def _supports_native_components(self) -> bool:
        """In-memory store only supports Polars components."""
        return False

    def _create_native_components(
        self,
    ) -> tuple[
        UpstreamJoiner[pl.LazyFrame],
        DataVersionCalculator[pl.LazyFrame],
        MetadataDiffResolver[pl.LazyFrame],
    ]:
        """Not supported - in-memory store only uses Polars components."""
        raise NotImplementedError(
            "InMemoryMetadataStore does not support native components"
        )

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

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop all metadata for a feature from in-memory storage.

        Args:
            feature_key: Feature key to drop metadata for
        """
        storage_key = self._get_storage_key(feature_key)

        # Remove from storage if it exists
        if storage_key in self._storage:
            del self._storage[storage_key]

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

        Raises:
            StoreNotOpenError: If store is not open
        """
        self._check_open()

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

    # ========== Context Manager Implementation ==========

    def open(self) -> None:
        """Open the in-memory store.

        For InMemoryMetadataStore, this is a no-op since no external
        resources need initialization.
        """
        pass  # No resources to initialize for in-memory storage

    def close(self) -> None:
        """Close the in-memory store.

        For InMemoryMetadataStore, this is a no-op since no external
        resources need cleanup.
        """
        pass  # No resources to cleanup for in-memory storage

    def __repr__(self) -> str:
        """String representation."""
        num_features = len(self._storage)
        num_fallbacks = len(self.fallback_stores)
        return (
            f"InMemoryMetadataStore("
            f"features={num_features}, "
            f"fallback_stores={num_fallbacks})"
        )

    def display(self) -> str:
        """Display string for this store."""
        if self._is_open:
            num_features = len(self._storage)
            return f"InMemoryMetadataStore(features={num_features})"
        else:
            return "InMemoryMetadataStore()"

    # ========== Backend Reference Conversion ==========

    def _dataframe_to_ref(self, df: pl.DataFrame) -> pl.LazyFrame:
        """Convert DataFrame to LazyFrame reference.

        Args:
            df: Polars DataFrame

        Returns:
            LazyFrame (no data movement - lazy evaluation)
        """
        return df.lazy()

    def _feature_to_ref(self, feature: FeatureKey | type[Feature]) -> pl.LazyFrame:
        """Convert feature to LazyFrame reference.

        Args:
            feature: Feature to convert

        Returns:
            LazyFrame from stored DataFrame
        """
        from metaxy.metadata_store.exceptions import FeatureNotFoundError

        df = self._read_metadata_local(feature)
        if df is None:
            feature_key = self._resolve_feature_key(feature)
            raise FeatureNotFoundError(
                f"Feature {feature_key.to_string()} not found in store"
            )
        return df.lazy()

    def _sample_to_ref(self, sample_df: pl.DataFrame) -> pl.LazyFrame:
        """Convert sample DataFrame to LazyFrame.

        Args:
            sample_df: Input sample DataFrame

        Returns:
            LazyFrame (no data movement - lazy evaluation)
        """
        return sample_df.lazy()

    def _result_to_dataframe(self, result: pl.LazyFrame) -> pl.DataFrame:
        """Convert LazyFrame result to DataFrame.

        Args:
            result: LazyFrame with data_version column

        Returns:
            Collected DataFrame ready to write
        """
        return result.collect()
