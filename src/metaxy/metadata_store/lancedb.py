"""LanceDB metadata store implementation."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import logging

import narwhals as nw
import polars as pl

from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.models.types import FeatureKey
from metaxy.provenance.types import HashAlgorithm

logger = logging.getLogger(__name__)


class LanceDBMetadataStore(MetadataStore):
    """
    [LanceDB](https://lancedb.github.io/lancedb/) metadata store for vector and structured data.

    LanceDB is a columnar database optimized for vector search and multimodal data.
    Each feature is stored in its own Lance table within the database directory.

    Storage layout:
    - Each feature gets its own table: {namespace}__{feature_name}
    - System tables: metaxy-system__feature_versions, metaxy-system__migration_events
    - Tables are stored as Lance format files in the database_path directory

    Note: Uses Polars components for data processing (no native SQL execution).

    Example: Local Directory
        ```py
        from pathlib import Path
        store = LanceDBMetadataStore(Path("./lancedb"))
        ```

    Example: With Hash Algorithm
        ```py
        store = LanceDBMetadataStore(
            "metadata.lance",
            hash_algorithm=HashAlgorithm.XXHASH64
        )
        ```

    Example: With Fallback Stores
        ```py
        # Local store with production fallback
        prod_store = LanceDBMetadataStore("s3://prod-bucket/metadata")
        dev_store = LanceDBMetadataStore(
            "./dev-metadata",
            fallback_stores=[prod_store]
        )
        ```
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        database_path: str | Path,
        *,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize [LanceDB](https://lancedb.github.io/lancedb/) metadata store.

        The database directory is created automatically if it doesn't exist (local paths only).
        Tables are created on-demand when features are first written.

        Args:
            database_path: Directory path or URI for LanceDB tables. Can be:
                - Local path: `"./metadata"` or `Path("./metadata")`
                - S3 URI: `"s3://bucket/path/to/db"` (requires AWS credentials)
                - LanceDB Cloud: `"db://database-name"` (requires API key via LANCEDB_API_KEY env var)
                - Remote URI: Any URI supported by LanceDB (http://, https://, gs://, etc.)
            fallback_stores: Ordered list of read-only fallback stores.
                Used when upstream features are not in this store.
            **kwargs: Passed to [metaxy.metadata_store.base.MetadataStore][]
                (e.g., hash_algorithm, hash_truncation_length, prefer_native)

        Note:
            Unlike SQL stores, LanceDB doesn't require explicit table creation.
            Tables are created automatically when writing metadata.

        Example:
            ```py
            # Local directory
            store = LanceDBMetadataStore("./metadata")

            # With custom hash algorithm
            store = LanceDBMetadataStore(
                "./metadata",
                hash_algorithm=HashAlgorithm.SHA256
            )

            # S3 backend (requires AWS credentials)
            prod = LanceDBMetadataStore("s3://prod-bucket/metadata")

            # LanceDB Cloud (requires LANCEDB_API_KEY environment variable)
            cloud = LanceDBMetadataStore("db://my-database")

            # Local with fallback to production
            dev = LanceDBMetadataStore("./dev", fallback_stores=[prod])
            ```
        """
        # Store as string to handle both local paths and remote URIs
        self.database_path = str(database_path)
        self._conn: Any | None = None
        super().__init__(fallback_stores=fallback_stores, **kwargs)

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Default hash algorithm."""
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """LanceDB store uses Polars components."""
        return False

    def native_implementation(self) -> nw.Implementation:
        """LanceDB operations run via Polars/Narwhals."""
        return nw.Implementation.POLARS

    @contextmanager
    def _create_provenance_tracker(self, plan):
        """Create a Polars provenance tracker for LanceDB."""
        from metaxy.provenance.polars import PolarsProvenanceTracker

        tracker = PolarsProvenanceTracker(plan=plan)
        try:
            yield tracker
        finally:
            # No cleanup needed for Polars tracker
            pass

    def _create_native_components(self) -> tuple[Any, Any, Any]:
        """Not supported - LanceDB relies on Polars fallback components."""
        raise NotImplementedError(
            "LanceDBMetadataStore does not support native field provenance calculations"
        )

    def open(self) -> None:
        """Open LanceDB connection.

        For local filesystem paths, creates the directory if it doesn't exist.
        For remote URIs (S3, LanceDB Cloud, etc.), connects directly.
        Tables are created on-demand when features are first written.

        Raises:
            ImportError: If lancedb package is not installed
            ConnectionError: If remote connection fails (e.g., invalid credentials)
        """
        import lancedb

        # Only create directory for local filesystem paths
        # Remote URIs (s3://, db://, http://, https://, gs://, etc.) are handled by LanceDB
        if not self._is_remote_uri(self.database_path):
            Path(self.database_path).mkdir(parents=True, exist_ok=True)

        self._conn = lancedb.connect(self.database_path)

    def close(self) -> None:
        """Close LanceDB connection.

        Releases resources by setting connection to None.
        Safe to call multiple times (idempotent).
        """
        self._conn = None

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

    @staticmethod
    def _is_remote_uri(path: str) -> bool:
        """Check if path is a remote URI rather than local filesystem path.

        Args:
            path: Database path or URI

        Returns:
            True if path is a remote URI (s3://, db://, http://, etc.)
        """
        remote_prefixes = (
            "s3://",  # S3 and S3-compatible storage
            "db://",  # LanceDB Cloud
            "http://",  # HTTP remote
            "https://",  # HTTPS remote
            "gs://",  # Google Cloud Storage
            "az://",  # Azure Blob Storage
            "file://",  # Explicit file URI
        )
        return path.startswith(remote_prefixes)

    def _table_name(self, feature_key: FeatureKey) -> str:
        return feature_key.table_name

    def _table_exists(self, table_name: str) -> bool:
        assert self._conn is not None, "Store must be open"
        return table_name in self._conn.table_names()  # type: ignore[attr-defined]

    def _get_table(self, table_name: str):
        assert self._conn is not None, "Store must be open"
        return self._conn.open_table(table_name)  # type: ignore[attr-defined]

    # Storage ------------------------------------------------------------------

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """Append metadata to Lance table.

        Creates the table if it doesn't exist, otherwise appends to existing table.
        Uses Arrow format for efficient storage.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated by base class)
        """
        assert self._conn is not None, "Store must be open"
        table_name = self._table_name(feature_key)

        arrow_table = df.to_arrow()

        if self._table_exists(table_name):
            table = self._get_table(table_name)
            table.add(arrow_table)  # type: ignore[attr-defined]
        else:
            self._conn.create_table(table_name, data=arrow_table)  # type: ignore[attr-defined]

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Lance table for feature.

        Permanently removes the Lance table from the database directory.
        Safe to call even if table doesn't exist (no-op).

        Args:
            feature_key: Feature key to drop metadata for
        """
        assert self._conn is not None, "Store must be open"
        table_name = self._table_name(feature_key)
        if self._table_exists(table_name):
            self._conn.drop_table(table_name)  # type: ignore[attr-defined]

    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from Lance table.

        Loads data from Lance, converts to Polars, and returns as Narwhals LazyFrame.
        Applies filters and column selection in memory.

        Args:
            feature: Feature to read
            feature_version: Filter by specific feature_version
            filters: List of Narwhals filter expressions
            columns: Optional list of columns to select

        Returns:
            Narwhals LazyFrame with metadata, or None if table not found
        """
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        table_name = self._table_name(feature_key)
        if not self._table_exists(table_name):
            return None

        table = self._get_table(table_name)

        try:
            pl_lazy = table.to_polars()
            # Smoke-test the LazyFrame to surface the batch_size issue eagerly.
            pl_lazy.limit(0).collect()
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            logger.debug("Polars/LanceDB batch_size incompatibility hit; converting via Arrow ")
            # Fall back to eager Arrow conversion until LanceDB issue #1539 is resolved.
            arrow_table = table.to_arrow()
            pl_lazy = pl.DataFrame(arrow_table).lazy()

        nw_lazy = nw.from_native(pl_lazy)

        if feature_version is not None:
            nw_lazy = nw_lazy.filter(
                nw.col("metaxy_feature_version") == feature_version
            )

        if filters is not None:
            for expr in filters:
                nw_lazy = nw_lazy.filter(expr)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def _list_features_local(self) -> list[FeatureKey]:
        """List Lance tables stored locally (excluding system tables)."""
        if self._conn is None:
            return []
        names = self._conn.table_names()  # type: ignore[attr-defined]

        features = []
        for name in names:
            feature_key = FeatureKey(name.split("__"))
            # Skip system tables
            if not self._is_system_table(feature_key):
                features.append(feature_key)

        return sorted(features)

    # Display ------------------------------------------------------------------

    def display(self) -> str:
        """Human-readable representation."""
        details = [f"path={self.database_path}"]
        if self._is_open:
            details.append(f"features={len(self._list_features_local())}")
        return f"LanceDBMetadataStore({', '.join(details)})"
