"""LanceDB storage handler using lancedb and Polars."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.polars.storage_config import PolarsStorageConfig
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.storage_handler import StorageHandler
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.types import FeatureKey


class LanceDBHandler(StorageHandler[None]):
    """Reads and writes LanceDB tables via the lancedb Python client.

    The handler manages its own LanceDB connection independently of the
    compute engine, using lifecycle hooks to open and close it.
    """

    def __init__(
        self,
        uri: str,
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._uri = str(uri)
        self._connect_kwargs: dict[str, Any] = connect_kwargs or {}
        self._lancedb_conn: Any = None

    # -- Lifecycle hooks -----------------------------------------------------------

    def on_connection_opened(self, conn: None) -> None:
        import lancedb

        if is_local_path(self._uri):
            Path(self._uri).mkdir(parents=True, exist_ok=True)
        self._lancedb_conn = lancedb.connect(self._uri, **self._connect_kwargs)

    def on_connection_closing(self) -> None:
        self._lancedb_conn = None

    # -- StorageHandler interface --------------------------------------------------

    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        return isinstance(storage_config, PolarsStorageConfig)

    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        return isinstance(storage_config, PolarsStorageConfig)

    def read(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        table_name = self._table_name(key)
        if not self._table_exists(table_name):
            return None
        nw_lazy = nw.from_native(self._get_table(table_name).to_polars())
        if filters:
            nw_lazy = nw_lazy.filter(*filters)
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)
        return nw_lazy

    def write(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        table_name = self._table_name(key)
        df_polars = collect_to_polars(df)
        if self._table_exists(table_name):
            self._get_table(table_name).add(df_polars)
        else:
            self._conn.create_table(table_name, data=df_polars)

    def has_feature(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> bool:
        return self._table_exists(self._table_name(key))

    def drop(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> None:
        table_name = self._table_name(key)
        if self._table_exists(table_name):
            self._conn.drop_table(table_name)

    def delete(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        table_name = self._table_name(key)
        table = self._get_table(table_name)

        if not filters:
            table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate, unquote_identifiers

        lf = self.read(None, storage_config, key)
        if lf is None:
            return
        table.delete(
            narwhals_expr_to_sql_predicate(
                nw.all_horizontal(list(filters), ignore_nulls=False),
                lf.collect_schema(),
                dialect="datafusion",
                extra_transforms=unquote_identifiers(),
            )
        )

    # -- Internal helpers ----------------------------------------------------------

    @property
    def _conn(self) -> Any:
        if self._lancedb_conn is None:
            raise RuntimeError("LanceDB connection is not open.")
        return self._lancedb_conn

    def _table_name(self, key: FeatureKey) -> str:
        return key.table_name

    def _table_exists(self, table_name: str) -> bool:
        # LanceDB has no existence check API; open_table raises on missing tables.
        try:
            self._conn.open_table(table_name)
        except (ValueError, FileNotFoundError):
            return False
        return True

    def _get_table(self, table_name: str) -> Any:
        return self._conn.open_table(table_name)
