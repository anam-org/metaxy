"""LanceDB metadata store implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy._decorators import public
from metaxy._utils import collect_to_polars
from metaxy.ext.lancedb.config import LanceDBMetadataStoreConfig
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path, sanitize_uri
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

logger = logging.getLogger(__name__)


@public
class LanceDBMetadataStore(MetadataStore):
    """
    [LanceDB](https://lancedb.github.io/lancedb/) metadata store for vector and structured data.

    LanceDB is a columnar database optimized for vector search and multimodal data.
    Each feature is stored in its own Lance table within the database directory.
    Uses Polars components for data processing (no native SQL execution).

    Example: Local Directory
        ```py
        from pathlib import Path
        from metaxy.ext.metadata_stores.lancedb import LanceDBMetadataStore

        store = LanceDBMetadataStore(Path("/path/to/featuregraph"))
        ```

    Example: Object Storage (S3, GCS, Azure)
        ```py
        store = LanceDBMetadataStore("s3:///path/to/featuregraph")
        ```

    Example: LanceDB Cloud
        ```py
        import os

        os.environ["LANCEDB_API_KEY"] = "your-api-key"
        store = LanceDBMetadataStore("db://my-database")
        ```
    """

    _should_warn_auto_create_tables = False
    versioning_engine_cls = PolarsVersioningEngine

    uri: str
    _conn: Any
    _connect_kwargs: dict[str, Any]

    def __init__(
        self,
        uri: str | Path | None = None,
        *,
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        connect_kwargs: dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    def __new__(
        cls,
        uri: str | Path | None = None,
        *,
        fallback_stores: list[MetadataStore] | None = None,
        connect_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LanceDBMetadataStore:
        if uri is None:
            raise ValueError("uri is required")

        from metaxy.metadata_store.compute_engine import PolarsComputeEngine

        instance = MetadataStore.__new__(cls)
        instance.uri = str(uri)
        instance._conn = None
        instance._connect_kwargs = connect_kwargs or {}

        MetadataStore.__init__(
            instance,
            engine=PolarsComputeEngine(),
            fallback_stores=fallback_stores,
            auto_create_tables=True,
            **kwargs,
        )
        return instance

    @contextmanager
    def _create_versioning_engine(self, plan):
        engine = PolarsVersioningEngine(plan=plan)
        try:
            yield engine
        finally:
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
        from metaxy.metadata_store.exceptions import StoreNotOpenError

        if self._conn is None:
            raise StoreNotOpenError("LanceDB connection is not open. Store must be used as a context manager.")
        return self._conn

    # Helpers -----------------------------------------------------------------

    def _table_name(self, feature_key: FeatureKey) -> str:
        return feature_key.table_name

    def _table_exists(self, table_name: str) -> bool:
        # LanceDB has no existence check API; open_table raises on missing tables.
        try:
            self.conn.open_table(table_name)
            return True
        except (ValueError, FileNotFoundError):
            return False

    def _get_table(self, table_name: str):
        return self.conn.open_table(table_name)

    # ===== MetadataStore abstract methods =====

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        feature_key = self._resolve_feature_key(feature)
        return self._table_exists(self._table_name(feature_key))

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    # Storage ------------------------------------------------------------------

    def _write_feature(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        df_polars = collect_to_polars(df)
        table_name = self._table_name(feature_key)

        if self._table_exists(table_name):
            self._get_table(table_name).add(df_polars)
        else:
            self.conn.create_table(table_name, data=df_polars)

    def _drop_feature(self, feature_key: FeatureKey) -> None:
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
        table_name = self._table_name(feature_key)
        table = self._get_table(table_name)

        if not filters:
            table.delete()
            return

        combined_filter = nw.all_horizontal(list(filters), ignore_nulls=False)

        from metaxy.metadata_store.utils import (
            narwhals_expr_to_sql_predicate,
            unquote_identifiers,
        )

        schema = self.read_feature_schema_from_store(feature_key)
        table.delete(
            narwhals_expr_to_sql_predicate(
                combined_filter,
                schema,
                dialect="datafusion",
                extra_transforms=unquote_identifiers(),
            )
        )

    def _read_feature(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        table_name = self._table_name(feature_key)
        if not self._table_exists(table_name):
            return None

        nw_lazy = nw.from_native(self._get_table(table_name).to_polars())

        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    # Display ------------------------------------------------------------------

    def display(self) -> str:
        return f"LanceDBMetadataStore(path={sanitize_uri(self.uri)})"

    @classmethod
    def config_model(cls) -> type[LanceDBMetadataStoreConfig]:
        return LanceDBMetadataStoreConfig
