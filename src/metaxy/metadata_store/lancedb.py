"""LanceDB metadata store implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import narwhals as nw
from narwhals.typing import Frame
from pydantic import Field
from typing_extensions import Self

from metaxy._decorators import public
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


@public
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
            "db://my-database", connect_kwargs={"api_key": "your-api-key", "region": "us-east-1"}
        )
        ```
    """

    _should_warn_auto_create_tables = False
    versioning_engine_cls = PolarsVersioningEngine

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
            raise StoreNotOpenError("LanceDB connection is not open. Store must be used as a context manager.")
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
            self.conn.open_table(table_name)
            return True
        except (ValueError, FileNotFoundError):
            # LanceDB raises ValueError when table doesn't exist
            return False

    def _get_table(self, table_name: str):
        return self.conn.open_table(table_name)

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

        if self._table_exists(table_name):
            table = self._get_table(table_name)
            table.add(df_polars)
        else:
            self.conn.create_table(table_name, data=df_polars)

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Lance table for feature.

        Permanently removes the Lance table from the database directory.
        Safe to call even if table doesn't exist (no-op).

        Args:
            feature_key: Feature key to drop metadata for
        """
        table_name = self._table_name(feature_key)
        if self._table_exists(table_name):
            self.conn.drop_table(table_name)

    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        current_only: bool,
    ) -> None:
        """Hard deletion for LanceDB. Calls the delete method on the Lance table.

        Args:
            feature_key: Feature to delete from
            filter_expr: Narwhals expression to select rows to delete
        """
        table_name = self._table_name(feature_key)
        table = self._get_table(table_name)

        # If no filters provided, delete all rows
        if not filters:
            table.delete()
            return

        # LanceDB's delete() API requires a single SQL WHERE clause string,
        # so combine multiple filter expressions into one using all_horizontal
        combined_filter = nw.all_horizontal(list(filters), ignore_nulls=False)

        # Convert Narwhals expression to SQL WHERE clause for LanceDB.
        # LanceDB requires unquoted identifiers, so we use the unquote_identifiers transform.
        from metaxy.metadata_store.utils import (
            narwhals_expr_to_sql_predicate,
            unquote_identifiers,
        )

        schema = self.read_feature_schema_from_store(feature_key)
        filter_str = narwhals_expr_to_sql_predicate(
            combined_filter,
            schema,
            dialect="datafusion",
            extra_transforms=unquote_identifiers(),
        )

        # Use native LanceDB delete for efficient in-place deletion
        table.delete(filter_str)

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
        # LanceDB's to_polars() returns a Polars LazyFrame directly
        # (fixed in Polars via https://github.com/pola-rs/polars/pull/25654)
        pl_lazy = table.to_polars()
        nw_lazy = nw.from_native(pl_lazy)

        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    # Display ------------------------------------------------------------------

    def display(self) -> str:
        """Human-readable representation with sanitized credentials."""
        path = sanitize_uri(self.uri)
        return f"LanceDBMetadataStore(path={path})"

    @classmethod
    def config_model(cls) -> type[LanceDBMetadataStoreConfig]:
        return LanceDBMetadataStoreConfig
