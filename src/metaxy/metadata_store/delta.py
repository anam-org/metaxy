"""Delta Lake metadata store implemented with delta-rs."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import narwhals as nw
import polars as pl

from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.exceptions import TableNotFoundError
from metaxy.models.feature import BaseFeature
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.provenance import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


class DeltaMetadataStore(MetadataStore):
    """
    Delta Lake metadata store backed by [delta-rs](https://github.com/delta-io/delta-rs).

    Stores each feature's metadata in a dedicated Delta table located under ``root_path``.
    Polars is used to write the tables. Supports configurable storage
    layouts (flat vs. nested directories), and forwards ``delta_write_options`` to every
    delta-rs write call.

    **Lazy Frame Handling:**

    - **Eager DataFrames**: Written directly using Polars' native write_delta

    - **Lazy Frames with streaming_chunk_size=None** (default): Collected once then written

    - **Lazy Frames with streaming_chunk_size=N**: Streamed in batches of size N without full collection,
      enabling processing of datasets larger than RAM (at the cost of slower performance and creating multiple DeltaLake versions within a single sink operation)

    Example: Standard Usage
        ```py
        # Standard usage (collects lazy frames before writing)
        store = DeltaMetadataStore("/data/metaxy/metadata")

        with store:
            store.write_metadata(MyFeature, metadata_df)
        ```

    Example: Providing Object Storage options
        ```py
        store = DeltaMetadataStore(
            "/data/metaxy/metadata",
            storage_options={"AWS_REGION": "us-west-2"},
        )
        ```

    Example: Sinking Lazy Frames
        ```py
        # Streaming mode for large datasets
        store = DeltaMetadataStore(
            "/data/metaxy/metadata",
            streaming_chunk_size=10000,  # Stream in 10k row batches
        )
        ```
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        root_path: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        streaming_chunk_size: int | None = None,
        layout: Literal["flat", "nested"] = "flat",
        delta_write_options: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Delta Lake metadata store.

        Args:
            root_path: Base directory or URI where feature tables are stored.
                Supports local paths (/path/to/dir), s3:// URLs, and other object store URIs.
            storage_options: Storage backend options passed to delta-rs.
                Example: {"AWS_REGION": "us-west-2", "AWS_ACCESS_KEY_ID": "...", ...}
                For S3: AWS_* environment variables
                For Azure: AZURE_* environment variables
                For GCS: GOOGLE_* environment variables
            streaming_chunk_size: Optional chunk size for streaming writes of lazy frames.
                - **None (default)**: Collect lazy frames fully before writing (fast, simple, higher memory)

                - **int (e.g., 10000)**: Stream lazy frames in batches of this size without full collection
                  (lower memory for large datasets, but slower and creates more DeltaLake versions)

                Trade-offs when streaming:

                - ✓ Lower memory usage for datasets larger than RAM

                - ✗ Slower performance (Polars sink_batches overhead)

                - ✗ More DeltaLake versions (one per batch)

                - ✗ Uses unstable Polars API

                For most use cases, leave as None (default).
            layout: Directory layout used when deriving table paths from feature keys.
                - **"flat"** (default): `<namespace>__<feature>` directories

                - **"nested"**: `namespace/feature/...` directories
            delta_write_options: Extra keyword arguments forwarded to every
                `deltalake.write_deltalake` call and to Polars `write_delta`.
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][].
        """
        self.storage_options = storage_options or {}
        self.streaming_chunk_size = streaming_chunk_size
        self.layout: Literal["flat", "nested"] = layout
        write_opts: dict[str, Any] = {"schema_mode": "merge"}
        if delta_write_options:
            write_opts.update(delta_write_options)
        self._delta_write_options = write_opts
        self._delta_table_cache: dict[tuple[str, bool], Any] = {}

        root_str = str(root_path)
        is_remote = "://" in root_str and not root_str.startswith(
            ("file://", "local://")
        )
        self.root_path = root_str if is_remote else root_str.split("://", 1)[-1]
        super().__init__(fallback_stores=fallback_stores, **kwargs)

    # ===== MetadataStore abstract methods =====

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH64 by default to match other non-SQL stores."""
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """DeltaLake store relies on Polars components for provenance calculations."""
        return False

    def native_implementation(self) -> nw.Implementation:
        """Get native implementation for Delta store."""
        return nw.Implementation.POLARS

    @contextmanager
    def _create_provenance_tracker(
        self, plan: FeaturePlan
    ) -> Iterator[ProvenanceTracker]:
        """Create Polars provenance tracker for Delta store.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            PolarsProvenanceTracker instance
        """
        from metaxy.provenance.polars import PolarsProvenanceTracker

        tracker = PolarsProvenanceTracker(plan=plan)

        try:
            yield tracker
        finally:
            # No cleanup needed for Polars tracker
            pass

    def open(self) -> None:
        """Delta-rs opens connections lazily; nothing to do up front."""
        return None

    def close(self) -> None:
        """No persistent resources to release."""
        # delta-rs is used in one-shot write/read calls, so nothing to close.
        pass

    def _relative_feature_path(self, feature_key: FeatureKey) -> str:
        """Return layout-specific relative path for a feature key."""
        if self.layout == "flat":
            return feature_key.table_name
        elif self.layout == "nested":
            return "/".join(feature_key.parts)
        else:
            raise ValueError(f"Unknown layout: {self.layout}")

    def _relative_path_to_feature_key(self, relative_path: Path) -> FeatureKey:
        """Convert a relative directory path back into a FeatureKey."""
        if self.layout == "flat":
            parts = relative_path.parts
            if len(parts) != 1:
                raise ValueError(
                    f"Flat layout expects single-segment directories, got {relative_path}"
                )
            return FeatureKey(parts[0].split("__"))
        elif self.layout == "nested":
            return FeatureKey(list(relative_path.parts))
        else:
            raise ValueError(f"Unknown layout: {self.layout}")

    @property
    def _deltalake(self):  # type: ignore[no-untyped-def]
        """Lazy import for delta-rs to avoid repeated module imports."""
        import deltalake  # pyright: ignore[reportMissingImports]

        return deltalake

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        """Return the URI/path used by deltalake for this feature."""

        relative = self._relative_feature_path(feature_key)
        return os.path.join(self.root_path, relative)

    def _table_exists(
        self, table_uri: str, *, feature_key: FeatureKey | None = None
    ) -> bool:
        """Check whether the provided URI already contains a Delta table."""
        if feature_key is not None:
            cache_key = feature_key.to_string()
            if (cache_key, True) in self._delta_table_cache or (
                cache_key,
                False,
            ) in self._delta_table_cache:
                return True

        return self._deltalake.DeltaTable.is_deltatable(
            table_uri,
            storage_options=self.storage_options or None,
        )

    def _get_delta_table(
        self, feature_key: FeatureKey, *, include_files: bool
    ):  # -> "deltalake.DeltaTable":
        """Return (and cache) a DeltaTable handle for the given feature."""
        cache_key = (feature_key.to_string(), include_files)
        table = self._delta_table_cache.get(cache_key)
        if table is None:
            table = self._deltalake.DeltaTable(
                self._feature_uri(feature_key),
                storage_options=self.storage_options or None,
                without_files=not include_files,
            )
            self._delta_table_cache[cache_key] = table
        return table

    def _invalidate_delta_table_cache(self, feature_key: FeatureKey) -> None:
        """Invalidate cached DeltaTable handles for this feature."""
        cache_key = feature_key.to_string()
        self._delta_table_cache.pop((cache_key, True), None)
        self._delta_table_cache.pop((cache_key, False), None)

    # ===== Storage operations =====

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame | pl.LazyFrame | nw.DataFrame[Any] | nw.LazyFrame[Any],
    ) -> None:
        """Append metadata to the Delta table for a feature.

        Handles both eager DataFrames and lazy LazyFrames:
        - **DataFrames**: Written directly via Polars write_delta
        - **LazyFrames**: Collected or streamed based on streaming_chunk_size setting

        Args:
            feature_key: Feature key to write to
            df: DataFrame or LazyFrame with metadata (already validated). Narwhals frames
                are accepted and converted to native Polars before writing.
        """
        # Convert Narwhals frames to native Polars
        if isinstance(df, (nw.DataFrame, nw.LazyFrame)):
            df = df.to_native()  # type: ignore[assignment]

        # Type narrowing: at this point df must be pl.DataFrame or pl.LazyFrame
        assert isinstance(df, (pl.DataFrame, pl.LazyFrame))

        table_uri = self._feature_uri(feature_key)
        table_exists = self._table_exists(table_uri, feature_key=feature_key)

        # Check if table exists
        if not table_exists and not self.auto_create_tables:
            raise TableNotFoundError(
                f"Delta table does not exist for feature {feature_key.to_string()} at {table_uri}."
            )

        # Delta automatically creates parent directories, no need to do it manually

        # Strategy: Use Polars write_delta for eager DataFrames (native support),
        # convert to PyArrow for lazy frames and use deltalake API (more flexible)
        if isinstance(df, pl.LazyFrame):
            # Check if streaming mode is enabled
            if self.streaming_chunk_size is not None:
                # Streaming mode: write batches incrementally without full collection
                # This enables processing datasets larger than RAM at the cost of:
                # - Slower performance (sink_batches overhead)
                # - Creating multiple DeltaLake versions within a single sink operation
                def write_batch(batch_df: pl.DataFrame) -> bool:
                    arrow_table = batch_df.to_arrow()
                    self._deltalake.write_deltalake(
                        table_uri,
                        arrow_table,
                        mode="append",
                        **self._delta_write_options,
                        storage_options=self.storage_options or None,
                    )
                    return False  # Continue processing all batches

                df.sink_batches(write_batch, chunk_size=self.streaming_chunk_size)  # pyright: ignore[reportCallIssue]
            else:
                # Default mode: collect then write (faster for normal-sized datasets)
                arrow_table = df.collect().to_arrow()
                self._deltalake.write_deltalake(
                    table_uri,
                    arrow_table,
                    mode="append",
                    **self._delta_write_options,
                    storage_options=self.storage_options or None,
                )
        else:
            # For eager DataFrames: use Polars native write_delta
            df.write_delta(
                table_uri,
                mode="append",
                delta_write_options=self._delta_write_options,
                storage_options=self.storage_options or None,
            )

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Delta table for the specified feature using soft delete.

        Uses Delta's delete operation which marks rows as deleted in the transaction log
        rather than physically removing files.
        """
        # Check if table exists first
        if not self._table_exists(
            self._feature_uri(feature_key), feature_key=feature_key
        ):
            return

        # Load the Delta table
        delta_table = self._get_delta_table(feature_key, include_files=False)

        # Use Delta's delete operation - soft delete all rows
        # This marks rows as deleted in transaction log without physically removing files
        delta_table.delete()

    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata stored in Delta for a single feature using lazy evaluation."""
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        table_uri = self._feature_uri(feature_key)
        if not self._table_exists(table_uri, feature_key=feature_key):
            return None

        # Use pl.scan_delta for lazy reading
        # Build column list for projection
        cols_to_read = None
        if columns is not None:
            cols_to_read = list(columns)
            # Ensure system columns are included for filtering
            if (
                feature_version is not None
                and "metaxy_feature_version" not in cols_to_read
            ):
                cols_to_read.append("metaxy_feature_version")

        # Use scan_delta for lazy evaluation
        lf = pl.scan_delta(
            table_uri,
            storage_options=self.storage_options or None,
        )

        # Apply column selection early if specified (pushdown)
        if cols_to_read is not None:
            lf = lf.select(cols_to_read)

        # Convert to Narwhals
        nw_lazy = nw.from_native(lf)
        return self._apply_read_filters(
            nw_lazy,
            feature_version=feature_version,
            filters=filters,
            columns=columns,
        )

    def _list_features_local(self) -> list[FeatureKey]:
        """List all features that have Delta tables in this store.

        Feature discovery is not supported for Delta stores. Features must be
        registered via 'metaxy graph push' or accessed directly by key.
        Use system tables (feature_versions) to query registered features.

        Returns:
            Empty list - feature discovery not supported
        """
        return []

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self.root_path}", f"layout={self.layout}"]
        if self.storage_options:
            details.append("storage_options=***")
        return f"DeltaMetadataStore({', '.join(details)})"
