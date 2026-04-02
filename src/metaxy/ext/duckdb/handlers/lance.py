"""DuckDB Lance storage handler — reads and writes Lance tables through DuckDB's lance extension."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy.metadata_store.io_handler import IOHandler
from metaxy.metadata_store.storage_config import LanceStorageConfig, StorageConfig
from metaxy.metadata_store.table_ref import LanceTableIdentifier

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend

from metaxy.ext.duckdb.engine import ExtensionSpec


class DuckDBLanceHandler(IOHandler["SQLBackend", LanceTableIdentifier]):
    """Reads and writes Lance tables through DuckDB's `lance` extension.

    Uses `__lance_scan()` for reads and `COPY TO (FORMAT LANCE)` for writes,
    keeping all I/O inside DuckDB.
    """

    def can_handle(self, storage_config: StorageConfig) -> bool:
        return isinstance(storage_config, LanceStorageConfig)

    def required_extensions(self) -> list[ExtensionSpec]:
        """DuckDB extensions this handler needs."""
        return [ExtensionSpec(name="lance", repository="community")]

    # -- CRUD -----------------------------------------------------------------

    def read(
        self,
        conn: SQLBackend,
        table_id: LanceTableIdentifier,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        uri = table_id.uri

        raw = conn.con  # ty: ignore[unresolved-attribute]
        try:
            raw.execute(f"SELECT 1 FROM __lance_scan('{uri}') LIMIT 0")
        except Exception:
            return None

        view_name = f"_lance_{table_id.table_name}"
        raw.execute(f"CREATE OR REPLACE TEMP VIEW \"{view_name}\" AS SELECT * FROM __lance_scan('{uri}')")

        ibis_table = conn.table(view_name)
        nw_frame: nw.LazyFrame[Any] = nw.from_native(ibis_table, eager_only=False)  # type: ignore[assignment]

        if filters:
            nw_frame = nw_frame.filter(*filters)  # ty: ignore[invalid-argument-type]
        if columns is not None:
            nw_frame = nw_frame.select(columns)
        return nw_frame

    def write(
        self,
        conn: SQLBackend,
        table_id: LanceTableIdentifier,
        df: Frame,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        uri = table_id.uri

        view_name = f"_lance_write_{table_id.table_name}"
        raw = conn.con  # ty: ignore[unresolved-attribute]
        raw.register(view_name, df.to_native())

        mode = "append" if self.has_feature(conn, table_id) else "create"
        raw.execute(f"""COPY "{view_name}" TO '{uri}' (FORMAT LANCE, mode '{mode}')""")

    def has_feature(
        self,
        conn: SQLBackend,
        table_id: LanceTableIdentifier,
    ) -> bool:
        raw = conn.con  # ty: ignore[unresolved-attribute]
        try:
            raw.execute(f"SELECT 1 FROM __lance_scan('{table_id.uri}') LIMIT 0")
            return True
        except Exception:
            return False

    def drop(
        self,
        conn: SQLBackend,
        table_id: LanceTableIdentifier,
    ) -> None:
        uri = table_id.uri
        if not uri.startswith(("s3://", "gs://", "az://", "http://", "https://")):
            import shutil
            from pathlib import Path as _Path

            path = _Path(uri)
            if path.exists():
                shutil.rmtree(path)

    def delete(
        self,
        conn: SQLBackend,  # noqa: ARG002
        table_id: LanceTableIdentifier,
        filters: Sequence[nw.Expr] | None,  # noqa: ARG002
        *,
        with_feature_history: bool,  # noqa: ARG002
    ) -> None:
        raise NotImplementedError("Row-level deletion not yet supported for DuckDB Lance handler.")

    def get_store_metadata(
        self,
        table_id: LanceTableIdentifier,
    ) -> dict[str, Any]:
        return {"uri": table_id.uri}
