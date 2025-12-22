"""In-memory metadata store implementation."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class InMemoryMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for InMemoryMetadataStore.

    Example:
        ```python
        config = InMemoryMetadataStoreConfig(
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

        store = InMemoryMetadataStore.from_config(config)
        ```
    """

    pass


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
    versioning_engine_cls = PolarsVersioningEngine

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

    @contextmanager
    def _create_versioning_engine(self, plan) -> Iterator[PolarsVersioningEngine]:
        """Create Polars provenance engine for in-memory store.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            PolarsVersioningEngine instance
        """
        from metaxy.versioning.polars import PolarsVersioningEngine

        # Create engine (only accepts plan parameter)
        engine = PolarsVersioningEngine(plan=plan)

        try:
            yield engine
        finally:
            # No cleanup needed for Polars engine
            pass

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        feature_key = self._resolve_feature_key(feature)
        storage_key = self._get_storage_key(feature_key)
        return storage_key in self._storage

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """
        Internal write implementation for in-memory storage.

        Args:
            feature_key: Feature key to write to
            df: Narwhals Frame (eager or lazy) with metadata (already validated)
            **kwargs: Backend-specific parameters (currently unused)
        """
        df_polars: pl.DataFrame = collect_to_polars(df)

        storage_key = self._get_storage_key(feature_key)

        # Append or create
        if storage_key in self._storage:
            existing_df = self._storage[storage_key]

            # Handle schema evolution: ensure both DataFrames have matching columns
            # Add missing columns as null to the existing DataFrame
            for col_name in df_polars.columns:
                if col_name not in existing_df.columns:
                    # Get the data type from the new DataFrame
                    col_dtype = df_polars.schema[col_name]
                    # Add column with null values of the appropriate type
                    existing_df = existing_df.with_columns(
                        pl.lit(None).cast(col_dtype).alias(col_name)
                    )

            # Add missing columns to the new DataFrame
            for col_name in existing_df.columns:
                if col_name not in df_polars.columns:
                    # Get the data type from the existing DataFrame
                    col_dtype = existing_df.schema[col_name]
                    # Add column with null values of the appropriate type
                    df_polars = df_polars.with_columns(
                        pl.lit(None).cast(col_dtype).alias(col_name)
                    )

            # Ensure column order matches by selecting columns in consistent order
            all_columns = sorted(set(existing_df.columns) | set(df_polars.columns))
            existing_df = existing_df.select(all_columns)
            df_polars = df_polars.select(all_columns)

            # Now we can safely concat
            self._storage[storage_key] = pl.concat(
                [existing_df, df_polars],
                how="vertical",
            )
        else:
            # Create new
            self._storage[storage_key] = df_polars

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
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """
        Read metadata from this store only (no fallback).

        Args:
            feature: Feature to read
            feature_version: Filter by specific feature_version
            filters: List of Narwhals filter expressions
            columns: Optional list of columns to select
            **kwargs: Backend-specific parameters (currently unused)

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
            nw_lazy = nw_lazy.filter(
                nw.col("metaxy_feature_version") == feature_version
            )

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

    def clear(self) -> None:
        """
        Clear all metadata from store.

        Useful for testing.
        """
        self._storage.clear()

    # ========== Context Manager Implementation ==========

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:
        """Open the in-memory store (no-op for in-memory, but accepts mode for consistency).

        Args:
            mode: Access mode (accepted for consistency but ignored).

        Yields:
            Self: The store instance
        """
        # Increment context depth to support nested contexts
        self._context_depth += 1

        try:
            # Only perform actual open on first entry
            if self._context_depth == 1:
                # No actual connection needed for in-memory
                # Mark store as open and validate
                self._is_open = True
                self._validate_after_open()

            yield self
        finally:
            # Decrement context depth
            self._context_depth -= 1

            # Only perform actual close on last exit
            if self._context_depth == 0:
                # Nothing to clean up
                self._is_open = False

    def __repr__(self) -> str:
        """String representation."""
        num_fallbacks = len(self.fallback_stores)
        status = "open" if self._is_open else "closed"
        return (
            f"InMemoryMetadataStore(status={status}, fallback_stores={num_fallbacks})"
        )

    def display(self) -> str:
        """Display string for this store."""
        status = "open" if self._is_open else "closed"
        return f"InMemoryMetadataStore(status={status})"

    @classmethod
    def config_model(cls) -> type[InMemoryMetadataStoreConfig]:
        return InMemoryMetadataStoreConfig
