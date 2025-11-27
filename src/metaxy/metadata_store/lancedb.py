"""LanceDB metadata store implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field
from typing_extensions import Self

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path, sanitize_uri
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

logger = logging.getLogger(__name__)


class LanceDBMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for LanceDBMetadataStore.

    Example:
        ```python
        config = LanceDBMetadataStoreConfig(
            uri="/path/to/featuregraph",
            connect_kwargs={"api_key": "your-api-key"},
        )

        store = LanceDBMetadataStore.from_config(config)
        ```
    """

    uri: str | Path = Field(
        description="Directory path or URI for LanceDB tables.",
    )
    connect_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra keyword arguments passed to lancedb.connect().",
    )


class LanceDBMetadataStore(MetadataStore):
    """
    [LanceDB](https://lancedb.github.io/lancedb/) metadata store for vector and structured data.

    LanceDB is a columnar database optimized for vector search and multimodal data.
    Each feature is stored in its own Lance table within the database directory.
    Uses Polars components for data processing (no native SQL execution).

    Storage layout:

    - Each feature gets its own table: `{namespace}__{feature_name}`

    - Tables are stored as Lance format in the directory specified by the URI

    - LanceDB handles schema evolution, transactions, and compaction automatically

    Example: Local Directory
        ```py
        from pathlib import Path
        from metaxy.metadata_store.lancedb import LanceDBMetadataStore

        # Local filesystem
        store = LanceDBMetadataStore(Path("/path/to/featuregraph"))
        ```

    Example: Object Storage (S3, GCS, Azure)
        ```py
        # object store (requires credentials)
        store = LanceDBMetadataStore("s3:///path/to/featuregraph")
        ```

    Example: LanceDB Cloud
        ```py
        import os

        # Option 1: Environment variable
        os.environ["LANCEDB_API_KEY"] = "your-api-key"
        store = LanceDBMetadataStore("db://my-database")

        # Option 2: Explicit credentials
        store = LanceDBMetadataStore(
            "db://my-database",
            connect_kwargs={"api_key": "your-api-key", "region": "us-east-1"}
        )
        ```
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        uri: str | Path,
        *,
        fallback_stores: list[MetadataStore] | None = None,
        connect_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize [LanceDB](https://lancedb.com/docs/) metadata store.

        The database directory is created automatically if it doesn't exist (local paths only).
        Tables are created on-demand when features are first written.

        Args:
            uri: Directory path or URI for LanceDB tables. Supports:

                - **Local path**: `"./metadata"` or `Path("/data/metaxy/lancedb")`

                - **Object stores**: `s3://`, `gs://`, `az://` (requires cloud credentials)

                - **LanceDB Cloud**: `"db://database-name"` (requires API key)

                - **Remote HTTP/HTTPS**: Any URI supported by LanceDB

            fallback_stores: Ordered list of read-only fallback stores.
                When reading features not found in this store, Metaxy searches
                fallback stores in order. Useful for local dev → staging → production chains.
            connect_kwargs: Extra keyword arguments passed directly to
                [lancedb.connect()](https://lancedb.github.io/lancedb/python/python/#lancedb.connect).
                Useful for LanceDB Cloud credentials (api_key, region) when you cannot
                rely on environment variables.
            **kwargs: Passed to [metaxy.metadata_store.base.MetadataStore][]
                (e.g., hash_algorithm, hash_truncation_length, prefer_native)

        Note:
            Unlike SQL stores, LanceDB doesn't require explicit table creation.
            Tables are created automatically when writing metadata.
        """
        self.uri: str = str(uri)
        self._conn: Any | None = None
        self._connect_kwargs = connect_kwargs or {}
        super().__init__(
            fallback_stores=fallback_stores,
            auto_create_tables=True,
            versioning_engine_cls=PolarsVersioningEngine,
            **kwargs,
        )

    @contextmanager
    def _create_versioning_engine(self, plan):
        """Create Polars versioning engine for LanceDB.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            PolarsVersioningEngine instance
        """
        engine = PolarsVersioningEngine(plan=plan)
        try:
            yield engine
        finally:
            # No cleanup needed for Polars engine
            pass

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:
        """Open LanceDB connection.

        For local filesystem paths, creates the directory if it doesn't exist.
        For remote URIs (S3, LanceDB Cloud, etc.), connects directly.
        Tables are created on-demand when features are first written.

        Args:
            mode: Access mode (READ or WRITE). Accepted for consistency but not used
                by LanceDB (LanceDB handles concurrent access internally).

        Yields:
            Self: The store instance

        Raises:
            ConnectionError: If remote connection fails (e.g., invalid credentials)
        """
        # Increment context depth to support nested contexts
        self._context_depth += 1

        try:
            # Only perform actual open on first entry
            if self._context_depth == 1:
                import lancedb

                if is_local_path(self.uri):
                    Path(self.uri).mkdir(parents=True, exist_ok=True)

                self._conn = lancedb.connect(self.uri, **self._connect_kwargs)
                self._is_open = True
                self._validate_after_open()

            yield self
        finally:
            # Decrement context depth
            self._context_depth -= 1

            # Only perform actual close on last exit
            if self._context_depth == 0:
                self._conn = None
                self._is_open = False

    @property
    def conn(self) -> Any:
        """Get LanceDB connection.

        Returns:
            Active LanceDB connection

        Raises:
            StoreNotOpenError: If store is not open
        """
        from metaxy.metadata_store.exceptions import StoreNotOpenError

        if self._conn is None:
            raise StoreNotOpenError(
                "LanceDB connection is not open. Store must be used as a context manager."
            )
        return self._conn

    # Helpers -----------------------------------------------------------------

    def _table_name(self, feature_key: FeatureKey) -> str:
        return feature_key.table_name

    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists without listing all tables.

        Uses open_table() which is more efficient than listing all tables,
        especially for remote storage (S3, GCS, etc.) where listing is expensive.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        try:
            self.conn.open_table(table_name)  # type: ignore[attr-defined]
            return True
        except (ValueError, FileNotFoundError):
            # LanceDB raises ValueError when table doesn't exist
            return False

    def _get_table(self, table_name: str):
        return self.conn.open_table(table_name)  # type: ignore[attr-defined]

    # ===== MetadataStore abstract methods =====

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Check if feature exists in LanceDB store.

        Args:
            feature: Feature to check

        Returns:
            True if feature exists, False otherwise
        """
        feature_key = self._resolve_feature_key(feature)
        table_name = self._table_name(feature_key)
        return self._table_exists(table_name)

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH64 by default to match other non-SQL stores."""
        return HashAlgorithm.XXHASH64

    # Storage ------------------------------------------------------------------

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Append metadata to Lance table.

        Creates the table if it doesn't exist, otherwise appends to existing table.
        Uses LanceDB's native Polars/Arrow integration for efficient storage.

        Args:
            feature_key: Feature key to write to
            df: Narwhals Frame with metadata (already validated by base class)
        """
        # Convert Narwhals frame to Polars DataFrame
        df_polars = collect_to_polars(df)

        table_name = self._table_name(feature_key)

        # LanceDB supports both Polars DataFrames and Arrow tables directly
        # Try Polars first (native integration), fall back to Arrow if needed
        try:
            if self._table_exists(table_name):
                table = self._get_table(table_name)
                # Use Polars DataFrame directly - LanceDB handles conversion
                table.add(df_polars)  # type: ignore[attr-defined]
            else:
                # Create table from Polars DataFrame - LanceDB handles schema
                self.conn.create_table(table_name, data=df_polars)  # type: ignore[attr-defined]
        except TypeError as exc:
            if not self._should_fallback_to_arrow(exc):
                raise
            # Defensive fallback: Modern LanceDB (>=0.3) accepts Polars DataFrames natively,
            # but fall back to Arrow if an older version or edge case doesn't support it.
            # This ensures compatibility across LanceDB versions.
            logger.debug("Falling back to Arrow format for LanceDB write: %s", exc)
            arrow_table = df_polars.to_arrow()
            if self._table_exists(table_name):
                table = self._get_table(table_name)
                table.add(arrow_table)  # type: ignore[attr-defined]
            else:
                self.conn.create_table(table_name, data=arrow_table)  # type: ignore[attr-defined]

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Lance table for feature.

        Permanently removes the Lance table from the database directory.
        Safe to call even if table doesn't exist (no-op).

        Args:
            feature_key: Feature key to drop metadata for
        """
        table_name = self._table_name(feature_key)
        if self._table_exists(table_name):
            self.conn.drop_table(table_name)  # type: ignore[attr-defined]

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from Lance table.

        Loads data from Lance, converts to Polars, and returns as Narwhals LazyFrame.
        Applies filters and column selection in memory.

        Args:
            feature: Feature to read
            filters: List of Narwhals filter expressions
            columns: Optional list of columns to select
            **kwargs: Backend-specific parameters (unused)

        Returns:
            Narwhals LazyFrame with metadata, or None if table not found
        """
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        table_name = self._table_name(feature_key)
        if not self._table_exists(table_name):
            return None

        table = self._get_table(table_name)
        # https://github.com/lancedb/lancedb/issues/1539
        # Fall back to eager Arrow conversion until LanceDB issue #1539 is resolved.
        arrow_table = table.to_arrow()
        pl_lazy = pl.DataFrame(arrow_table).lazy()
        nw_lazy = nw.from_native(pl_lazy)

        if filters is not None:
            nw_lazy = nw_lazy.filter(filters)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    @staticmethod
    def _should_fallback_to_arrow(exc: TypeError) -> bool:
        """Return True when TypeError likely originates from Polars support gaps."""
        message = str(exc).lower()
        polars_markers = ("polars", "dataframe", "lazyframe", "data frame")
        return any(marker in message for marker in polars_markers)

    # ========== Error Tracking Implementation ==========

    def write_errors_to_store(
        self,
        feature_key: FeatureKey,
        errors_df: Frame,
    ) -> None:
        """Write error records to LanceDB error table.

        Args:
            feature_key: Feature key to write errors for
            errors_df: Narwhals DataFrame with error records
        """
        # Convert to Polars
        df_polars = collect_to_polars(errors_df)

        # Get error table name
        error_table_name = self._get_error_table_name(feature_key)

        # Write to LanceDB (with same fallback logic as regular metadata)
        try:
            if self._table_exists(error_table_name):
                table = self._get_table(error_table_name)
                table.add(df_polars)  # type: ignore[attr-defined]
            else:
                self.conn.create_table(error_table_name, data=df_polars)  # type: ignore[attr-defined]
        except TypeError as exc:
            if not self._should_fallback_to_arrow(exc):
                raise
            # Fallback to Arrow format
            logger.debug(
                "Falling back to Arrow format for LanceDB error write: %s", exc
            )
            arrow_table = df_polars.to_arrow()
            if self._table_exists(error_table_name):
                table = self._get_table(error_table_name)
                table.add(arrow_table)  # type: ignore[attr-defined]
            else:
                self.conn.create_table(error_table_name, data=arrow_table)  # type: ignore[attr-defined]

    def read_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read error records from LanceDB error table.

        Args:
            feature_key: Feature key to read errors for
            filters: Optional Narwhals filter expressions to apply

        Returns:
            Narwhals LazyFrame with error records, or None if error table doesn't exist
        """
        # Get error table name
        error_table_name = self._get_error_table_name(feature_key)

        # Check if error table exists
        if not self._table_exists(error_table_name):
            return None

        # Read from LanceDB
        table = self._get_table(error_table_name)
        arrow_table = table.to_arrow()
        pl_lazy = pl.DataFrame(arrow_table).lazy()
        nw_lazy = nw.from_native(pl_lazy)

        # Apply filters
        if filters is not None:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        return nw_lazy

    def clear_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        sample_uids: Sequence[dict[str, Any]] | None = None,
        feature_version: str | None = None,
    ) -> None:
        """Clear error records from LanceDB error table.

        Args:
            feature_key: Feature key to clear errors for
            sample_uids: Optional list of sample ID dicts to clear
            feature_version: Optional feature version to clear
        """
        # Get error table name
        error_table_name = self._get_error_table_name(feature_key)

        # Check if error table exists
        if not self._table_exists(error_table_name):
            return  # No-op if table doesn't exist

        # If no filters, drop entire error table
        if sample_uids is None and feature_version is None:
            self.conn.drop_table(error_table_name)  # type: ignore[attr-defined]
            return

        # Otherwise, read-filter-rewrite strategy
        # (LanceDB doesn't have native DELETE with WHERE, like Delta does)
        table = self._get_table(error_table_name)
        arrow_table = table.to_arrow()
        errors_df = pl.DataFrame(arrow_table)

        # Build filter conditions for what to KEEP (inverse of what to clear)
        keep_conditions = []

        # Filter by feature_version: keep errors from OTHER versions
        if feature_version is not None:
            from metaxy.models.constants import METAXY_FEATURE_VERSION

            keep_conditions.append(pl.col(METAXY_FEATURE_VERSION) != feature_version)

        # Filter by sample_uids: keep errors from OTHER samples
        if sample_uids is not None and len(sample_uids) > 0:
            # Get id_columns from feature spec
            feature_spec = self._resolve_feature_plan(feature_key).feature
            id_cols = list(feature_spec.id_columns)

            # Build condition to exclude specified samples
            sample_match_conditions = []
            for uid_dict in sample_uids:
                # Build AND condition for this sample
                and_conditions = [
                    pl.col(col_name) == uid_dict[col_name] for col_name in id_cols
                ]
                # Combine with &
                if len(and_conditions) == 1:
                    sample_condition = and_conditions[0]
                else:
                    sample_condition = and_conditions[0]
                    for cond in and_conditions[1:]:
                        sample_condition = sample_condition & cond
                sample_match_conditions.append(sample_condition)

            # Combine all sample conditions with OR (matches any of the samples)
            if len(sample_match_conditions) == 1:
                matches_any_sample = sample_match_conditions[0]
            else:
                matches_any_sample = sample_match_conditions[0]
                for cond in sample_match_conditions[1:]:
                    matches_any_sample = matches_any_sample | cond

            # Keep rows that DON'T match any sample
            keep_conditions.append(~matches_any_sample)

        # Apply keep conditions
        if keep_conditions:
            # Combine all keep conditions with AND
            if len(keep_conditions) == 1:
                final_condition = keep_conditions[0]
            else:
                final_condition = keep_conditions[0]
                for cond in keep_conditions[1:]:
                    final_condition = final_condition & cond

            filtered_df = errors_df.filter(final_condition)

            # Drop table and recreate with filtered data
            self.conn.drop_table(error_table_name)  # type: ignore[attr-defined]

            # Only recreate if there are rows to keep
            if len(filtered_df) > 0:
                try:
                    self.conn.create_table(error_table_name, data=filtered_df)  # type: ignore[attr-defined]
                except TypeError as exc:
                    if not self._should_fallback_to_arrow(exc):
                        raise
                    # Fallback to Arrow
                    arrow_table = filtered_df.to_arrow()
                    self.conn.create_table(error_table_name, data=arrow_table)  # type: ignore[attr-defined]

    # Display ------------------------------------------------------------------

    def display(self) -> str:
        """Human-readable representation with sanitized credentials."""
        path = sanitize_uri(self.uri)
        return f"LanceDBMetadataStore(path={path})"

    @classmethod
    def config_model(cls) -> type[LanceDBMetadataStoreConfig]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return LanceDBMetadataStoreConfig
