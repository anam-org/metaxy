"""In-memory metadata store implementation."""

from collections.abc import Sequence
from typing import Any

import narwhals as nw
import polars as pl

from metaxy.data_versioning.calculators.base import ProvenanceByFieldCalculator
from metaxy.data_versioning.diff.base import MetadataDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import IDColumns
from metaxy.models.types import FeatureKey


class InMemoryMetadataStore(MetadataStore):
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

    Notes:
        Uses Narwhals LazyFrames (nw.LazyFrame) for all operations

    Components:
        Components are created on-demand in resolve_update().
        Uses Polars internally but exposes Narwhals interface.
        Only supports Polars components (no native backend).
    """

    # Disable auto_create_tables warning for in-memory store
    # (table creation concept doesn't apply to memory storage)
    _should_warn_auto_create_tables: bool = False

    def __init__(self, **kwargs: Any):
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
        UpstreamJoiner,
        ProvenanceByFieldCalculator,
        MetadataDiffResolver,
    ]:
        """Not supported - in-memory store only uses Polars components."""
        raise NotImplementedError(
            "InMemoryMetadataStore does not support native field provenance calculations"
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
            existing_df = self._storage[storage_key]

            # Handle schema evolution: ensure both DataFrames have matching columns
            # Add missing columns as null to the existing DataFrame
            for col_name in df.columns:
                if col_name not in existing_df.columns:
                    # Get the data type from the new DataFrame
                    col_dtype = df.schema[col_name]
                    # Add column with null values of the appropriate type
                    existing_df = existing_df.with_columns(
                        pl.lit(None).cast(col_dtype).alias(col_name)
                    )

            # Add missing columns to the new DataFrame (for backward compatibility)
            for col_name in existing_df.columns:
                if col_name not in df.columns:
                    # Get the data type from the existing DataFrame
                    col_dtype = existing_df.schema[col_name]
                    # Add column with null values of the appropriate type
                    df = df.with_columns(pl.lit(None).cast(col_dtype).alias(col_name))

            # Ensure column order matches by selecting columns in consistent order
            all_columns = sorted(set(existing_df.columns) | set(df.columns))
            existing_df = existing_df.select(all_columns)
            df = df.select(all_columns)

            # Now we can safely concat
            self._storage[storage_key] = pl.concat(
                [existing_df, df],
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

    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature[IDColumns]],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """
        Read metadata from this store only (no fallback).

        Args:
            feature: Feature to read
            feature_version: Filter by specific feature_version
            filters: List of Narwhals filter expressions
            columns: Optional list of columns to select

        Returns:
            Narwhals LazyFrame with metadata, or None if not found

        Raises:
            StoreNotOpenError: If store is not open
        """
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        storage_key = self._get_storage_key(feature_key)

        if storage_key not in self._storage:
            return None

        # Start with lazy Polars DataFrame, wrap with Narwhals
        df_lazy = self._storage[storage_key].lazy()
        nw_lazy = nw.from_native(df_lazy)

        # Apply feature_version filter
        if feature_version is not None:
            nw_lazy = nw_lazy.filter(nw.col("feature_version") == feature_version)

        # Apply generic Narwhals filters
        if filters is not None:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        # Select columns
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        # Check if result would be empty (we need to check the underlying frame)
        # For now, return the lazy frame - emptiness check happens when materializing
        return nw_lazy

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
        resources need initialization. The auto_create_tables setting
        has no effect for in-memory stores (no tables to create).
        """
        # No resources to initialize for in-memory storage
        pass

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
