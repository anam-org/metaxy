"""LanceDB metadata store implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._decorators import public
from metaxy.ext.polars.versioning import PolarsVersioningEngine
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path, sanitize_uri
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.utils import collect_to_polars
from metaxy.versioning.types import HashAlgorithm

logger = logging.getLogger(__name__)


@public
class LanceDBMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for LanceDBMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.lancedb.LanceDBMetadataStore"

        [stores.dev.config]
        uri = "/path/to/featuregraph"

        [stores.dev.config.connect_kwargs]
        api_key = "your-api-key"
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
        from metaxy.ext.lancedb import LanceDBMetadataStore

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

    def _open(self, mode: AccessMode) -> None:  # noqa: ARG002
        import lancedb

        if is_local_path(self.uri):
            Path(self.uri).mkdir(parents=True, exist_ok=True)

        self._conn = lancedb.connect(self.uri, **self._connect_kwargs)

    def _close(self) -> None:
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
        """Use XXHASH32 by default."""
        return HashAlgorithm.XXHASH32

    # Map compatibility --------------------------------------------------------
    # LanceDB has no native Map type (https://github.com/lance-format/lance/issues/2341),
    # so metaxy `Map` columns are stored as named Structs and reconstructed on read.

    def _maps_to_structs(self, df: pl.DataFrame, feature_key: FeatureKey) -> pl.DataFrame:
        """Decompose `Map` columns into named `Struct` columns Lance can store."""
        from metaxy.utils._arrow_map import convert_maps_to_structs
        from metaxy.utils.dataframes import find_map_columns

        map_columns = find_map_columns(nw.from_native(df))
        # Empty frames carry no keys, so fall back to the feature's field names to keep the schema stable.
        fallback = {col: self._map_field_names(feature_key, col) for col in map_columns}
        return convert_maps_to_structs(df, map_columns, fallback_field_names=fallback)

    @staticmethod
    def _structs_to_maps(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Reconstruct metaxy system `Map` columns from the named `Struct` columns stored in Lance."""
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
        from metaxy.utils._arrow_map import convert_structs_to_maps

        return convert_structs_to_maps(lf, columns=[METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD])

    def _map_field_names(self, feature_key: FeatureKey, column: str) -> list[str]:
        """Struct field names for a metaxy system `Map` column, taken from the active graph."""
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
        from metaxy.models.feature import FeatureGraph

        if column not in {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}:
            return []
        definition = FeatureGraph.get_active().feature_definitions_by_key.get(feature_key)
        if definition is None:
            return []
        return [f.key.to_struct_key() for f in definition.spec.fields]

    # Storage ------------------------------------------------------------------

    def _write_feature(
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
        df_polars = self._maps_to_structs(collect_to_polars(df), feature_key)

        table_name = self._table_name(feature_key)

        if self._table_exists(table_name):
            table = self._get_table(table_name)
            table.add(df_polars)
        else:
            self.conn.create_table(table_name, data=df_polars)

    def _drop_feature(self, feature_key: FeatureKey) -> None:
        """Drop Lance table for feature.

        Permanently removes the Lance table from the database directory.
        Safe to call even if table doesn't exist (no-op).

        Args:
            feature_key: Feature key to drop metadata for
        """
        table_name = self._table_name(feature_key)
        if self._table_exists(table_name):
            self.conn.drop_table(table_name)

    def _delete_feature(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
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

    def _read_feature(
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
        nw_lazy = nw.from_native(self._structs_to_maps(table.to_polars()))

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
