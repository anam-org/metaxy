"""LanceDB storage handler."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy.metadata_store.io_handler import IOHandler
from metaxy.metadata_store.storage_config import LanceStorageConfig, StorageConfig
from metaxy.metadata_store.table_ref import LanceTableIdentifier
from metaxy.metadata_store.utils import is_local_path
from metaxy.utils import collect_to_polars


class LanceDBHandler(IOHandler[None, LanceTableIdentifier]):
    """Reads and writes Lance tables with [LanceDB](https://lancedb.github.io/lancedb/) and Polars.

    Database connections are opened on first use and released when the engine closes.
    """

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}

    def can_handle(self, storage_config: StorageConfig) -> bool:
        return isinstance(storage_config, LanceStorageConfig)

    def connection(self, database: str, connect_kwargs: dict[str, Any]) -> Any:
        """Return the LanceDB connection to `database`, opening it on first use."""
        key = json.dumps([database, connect_kwargs], sort_keys=True, default=str)
        if key not in self._connections:
            import lancedb

            if is_local_path(database):
                Path(database).mkdir(parents=True, exist_ok=True)
            self._connections[key] = lancedb.connect(database, **connect_kwargs)
        return self._connections[key]

    def _database(self, table_id: LanceTableIdentifier) -> Any:
        return self.connection(table_id.database, table_id.connect_kwargs)

    def on_close(self, conn: None) -> None:  # noqa: ARG002
        self._connections.clear()

    def read(
        self,
        conn: None,
        table_id: LanceTableIdentifier,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        if not self.has_feature(conn, table_id):
            return None
        lazy = nw.from_native(self._database(table_id).open_table(table_id.table_name).to_polars())
        if filters:
            lazy = lazy.filter(*filters)
        if columns is not None:
            lazy = lazy.select(columns)
        return lazy

    def write(
        self,
        conn: None,
        table_id: LanceTableIdentifier,
        df: Frame,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        df_polars = collect_to_polars(df)
        if self.has_feature(conn, table_id):
            self._database(table_id).open_table(table_id.table_name).add(df_polars)
        else:
            self._database(table_id).create_table(table_id.table_name, data=df_polars)

    def has_feature(
        self,
        conn: None,  # noqa: ARG002
        table_id: LanceTableIdentifier,
    ) -> bool:
        # Opening the table is cheaper than listing all tables, especially on object storage.
        try:
            self._database(table_id).open_table(table_id.table_name)
        except (ValueError, FileNotFoundError):
            return False
        return True

    def drop(
        self,
        conn: None,
        table_id: LanceTableIdentifier,
    ) -> None:
        if self.has_feature(conn, table_id):
            self._database(table_id).drop_table(table_id.table_name)

    def delete(
        self,
        conn: None,
        table_id: LanceTableIdentifier,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,  # noqa: ARG002
    ) -> None:
        lazy = self.read(conn, table_id)
        if lazy is None:
            return
        table = self._database(table_id).open_table(table_id.table_name)
        if not filters:
            table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate, unquote_identifiers

        # LanceDB takes a single SQL `WHERE` clause with unquoted identifiers.
        table.delete(
            narwhals_expr_to_sql_predicate(
                nw.all_horizontal(list(filters), ignore_nulls=False),
                lazy.collect_schema(),
                dialect="datafusion",
                extra_transforms=unquote_identifiers(),
            )
        )
