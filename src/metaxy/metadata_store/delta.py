"""Delta Lake metadata store implemented with delta-rs."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

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
    Uses Polars/Narwhals components for metadata operations, supports configurable storage
    layouts (flat vs. nested directories), and forwards ``delta_write_options`` to every
    delta-rs write call.

    **Lazy Frame Handling:**

    - **Eager DataFrames**: Written directly using Polars' native write_delta (efficient)

    - **Lazy Frames with streaming_chunk_size=None** (default): Collected once then written

    - **Lazy Frames with streaming_chunk_size=N**: Streamed in batches of size N without full collection,
      enabling processing of datasets larger than RAM (at the cost of slower performance and more Delta versions)

    Example:
        ```py
        # Standard usage (collects lazy frames before writing)
        store = DeltaMetadataStore("/data/metaxy/metadata")

        # Streaming mode for large datasets
        store = DeltaMetadataStore(
            "/data/metaxy/metadata",
            streaming_chunk_size=10000,  # Stream in 10k row batches
            storage_options={"AWS_REGION": "us-west-2"},
        )

        with store:
            with store.allow_cross_project_writes():
                store.write_metadata(MyFeature, metadata_df)
        ```
    """

    _should_warn_auto_create_tables = False

    @cached_property
    def _deltalake(self):
        """Lazy import for delta-rs to avoid repeated module imports."""
        import deltalake  # pyright: ignore[reportMissingImports]

        return deltalake

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
                  (lower memory for large datasets, but slower and creates more Delta versions)

                Trade-offs when streaming:

                - ✓ Lower memory usage for datasets larger than RAM

                - ✗ Slower performance (Polars sink_batches overhead)

                - ✗ More Delta versions (one per batch)

                - ✗ Uses unstable Polars API

                For most use cases, leave as None (default).
            layout: Directory layout used when deriving table paths from feature keys.
                - **"flat"**: Current `<namespace>__<feature>` directories (backward compatible default)

                - **"nested"**: `namespace/feature/...` directories for easier navigation in consoles.
            delta_write_options: Extra keyword arguments forwarded to every
                `deltalake.write_deltalake` call and to Polars `write_delta`.
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][].
        """
        self.storage_options = storage_options or {}
        self.streaming_chunk_size = streaming_chunk_size
        if layout not in ("flat", "nested"):
            raise ValueError("layout must be either 'flat' or 'nested'")
        self.layout: Literal["flat", "nested"] = layout
        write_opts: dict[str, Any] = {"schema_mode": "merge"}
        if delta_write_options:
            write_opts.update(delta_write_options)
        self._delta_write_options = write_opts
        self._delta_table_cache: dict[tuple[str, bool], Any] = {}
        self._display_root = str(root_path)

        root_str = str(root_path)
        parsed = urlsplit(root_str) if "://" in root_str else None
        scheme = parsed.scheme.lower() if parsed else ""
        local_schemes = {"", "file", "local"}

        if scheme in local_schemes:
            self._is_remote = False
            if scheme == "":
                local_path = Path(root_path).expanduser().resolve()
            else:
                assert parsed is not None
                authority = parsed.netloc
                path = parsed.path
                if scheme == "file":
                    raw_path = path or "/"
                else:
                    trimmed_path = path.lstrip("/") if authority else path
                    if authority and trimmed_path:
                        raw_path = f"{authority}/{trimmed_path}"
                    elif authority:
                        raw_path = authority
                    else:
                        raw_path = trimmed_path
                local_path = Path(raw_path or ".").expanduser().resolve()
            self._local_root_path = local_path
            self._root_uri = str(local_path)
        else:
            self._is_remote = bool(parsed and parsed.scheme)
            self._local_root_path = None
            self._root_uri = root_str.rstrip("/")

        self.root_path = self._local_root_path
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
        return "/".join(feature_key.parts)

    def _relative_path_to_feature_key(self, relative_path: Path) -> FeatureKey:
        """Convert a relative directory path back into a FeatureKey."""
        if self.layout == "flat":
            parts = relative_path.parts
            if len(parts) != 1:
                raise ValueError(
                    f"Flat layout expects single-segment directories, got {relative_path}"
                )
            return FeatureKey(parts[0].split("__"))
        return FeatureKey(list(relative_path.parts))

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        """Return the URI/path used by deltalake for this feature."""
        relative = self._relative_feature_path(feature_key)
        if self._local_root_path is not None:
            return str(self._local_root_path / Path(relative))
        base = self._root_uri.rstrip("/")
        return f"{base}/{relative}".rstrip("/")

    def _feature_local_path(self, feature_key: FeatureKey) -> Path | None:
        """Return filesystem path when operating on local roots."""
        if self._local_root_path is None:
            return None
        return self._local_root_path / Path(self._relative_feature_path(feature_key))

    def _table_exists(
        self, table_uri: str, *, feature_key: FeatureKey | None = None
    ) -> bool:
        """Check whether the provided URI already contains a Delta table.

        Works for both local and remote (object store) paths.
        """
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
        - **DataFrames**: Written directly via Polars write_delta (efficient)
        - **LazyFrames**: Collected or streamed based on streaming_chunk_size setting

        Args:
            feature_key: Feature key to write to
            df: DataFrame or LazyFrame with metadata (already validated). Narwhals frames
                are accepted and converted to native Polars before writing.

        Raises:
            TableNotFoundError: If table doesn't exist and auto_create_tables is False
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
                f"Delta table does not exist for feature {feature_key.to_string()} at {table_uri}. "
                f"Enable auto_create_tables=True to automatically create tables."
            )

        # Delta automatically creates parent directories, no need to do it manually
        self._invalidate_delta_table_cache(feature_key)

        # Strategy: Use Polars write_delta for eager DataFrames (native support),
        # convert to PyArrow for lazy frames and use deltalake API (more flexible)
        if isinstance(df, pl.LazyFrame):
            # Check if streaming mode is enabled
            if self.streaming_chunk_size is not None:
                # Streaming mode: write batches incrementally without full collection
                # This enables processing datasets larger than RAM at the cost of:
                # - Slower performance (sink_batches overhead)
                # - More Delta versions (one per batch)
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
            # This is efficient and leverages Polars' built-in Delta support
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
        self._invalidate_delta_table_cache(feature_key)

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

        Returns:
            List of FeatureKey objects (excluding system tables)
        """
        if self._local_root_path is not None:
            return self._list_features_from_local_root()
        return self._list_features_from_object_store()

    def _list_features_from_local_root(self) -> list[FeatureKey]:
        if self._local_root_path is None or not self._local_root_path.exists():
            return []

        feature_keys: list[FeatureKey] = []
        for delta_log_dir in self._local_root_path.rglob("_delta_log"):
            feature_dir = delta_log_dir.parent
            try:
                relative = feature_dir.relative_to(self._local_root_path)
                feature_key = self._relative_path_to_feature_key(relative)
            except ValueError as exc:
                warnings.warn(
                    f"Could not parse Delta directory '{feature_dir}' as FeatureKey: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if not self._is_system_table(feature_key):
                feature_keys.append(feature_key)
        return sorted(feature_keys)

    def _list_features_from_object_store(self) -> list[FeatureKey]:
        """List features from object store (S3, Azure, GCS, etc.).

        Note: For remote stores, feature discovery relies on the system tables.
        Use `metaxy graph push` to register features in the metadata store.

        Returns:
            Empty list - remote stores require explicit feature registration
        """
        warnings.warn(
            f"Feature discovery not supported for remote Delta stores ({self._root_uri}). "
            "Features must be registered via 'metaxy graph push' or accessed directly by key. "
            "Use system tables (feature_versions) to query registered features.",
            UserWarning,
            stacklevel=2,
        )
        return []

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self._display_root}", f"layout={self.layout}"]
        if self.storage_options:
            details.append("storage_options=***")
        return f"DeltaMetadataStore({', '.join(details)})"
