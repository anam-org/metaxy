"""DuckDB + Lance storage handler.

Reads and writes Lance tables through DuckDB's `lance` community extension.
Uses `__lance_scan()` for reads and `COPY TO (FORMAT LANCE)` for writes.

```python
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.storage_config import LanceStorageConfig
from metaxy.ext.metadata_stores.duckdb import DuckDBEngine, ExtensionSpec

handler = DuckDBLanceHandler()
engine = DuckDBEngine(
    database=":memory:",
    handler=handler,
    extensions=[*handler.required_extensions()],
)
store = MetadataStore(
    engine=engine,
    storage=[LanceStorageConfig(format="lance", location="/data/lance")],
)
```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy.ext.metadata_stores.duckdb import ExtensionSpec
from metaxy.metadata_store.storage_config import LanceStorageConfig, StorageConfig
from metaxy.metadata_store.storage_handler import StorageHandler
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend


class DuckDBLanceHandler(StorageHandler["SQLBackend"]):
    """Reads and writes Lance tables through DuckDB's `lance` extension.

    Uses `__lance_scan()` for reads and `COPY TO (FORMAT LANCE)` for writes,
    keeping all I/O inside DuckDB.
    """

    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, LanceStorageConfig)

    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, LanceStorageConfig)

    def required_extensions(self) -> list[ExtensionSpec]:
        """DuckDB extensions this handler needs."""
        return [ExtensionSpec(name="lance", repository="community")]

    # -- helpers --------------------------------------------------------------

    def _table_uri(self, storage_config: LanceStorageConfig, key: FeatureKey) -> str:
        location = storage_config.location.rstrip("/")
        return f"{location}/{key.table_name}.lance"

    # -- CRUD -----------------------------------------------------------------

    def read(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        assert isinstance(storage_config, LanceStorageConfig)
        uri = self._table_uri(storage_config, key)

        raw = conn.con  # ty: ignore[unresolved-attribute]
        try:
            raw.execute(f"SELECT 1 FROM __lance_scan('{uri}') LIMIT 0")
        except Exception:
            return None

        view_name = f"_lance_{key.table_name}"
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
        storage_config: StorageConfig,
        key: FeatureKey,
        df: Frame,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        assert isinstance(storage_config, LanceStorageConfig)
        uri = self._table_uri(storage_config, key)

        view_name = f"_lance_write_{key.table_name}"
        raw = conn.con  # ty: ignore[unresolved-attribute]
        raw.register(view_name, df.to_native())

        mode = "append" if self.has_feature(conn, storage_config, key) else "create"
        raw.execute(f"""COPY "{view_name}" TO '{uri}' (FORMAT LANCE, mode '{mode}')""")

    def has_feature(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> bool:
        assert isinstance(storage_config, LanceStorageConfig)
        uri = self._table_uri(storage_config, key)
        raw = conn.con  # ty: ignore[unresolved-attribute]
        try:
            raw.execute(f"SELECT 1 FROM __lance_scan('{uri}') LIMIT 0")
            return True
        except Exception:
            return False

    def drop(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> None:
        assert isinstance(storage_config, LanceStorageConfig)
        # Lance files are directories — remove via filesystem
        uri = self._table_uri(storage_config, key)
        if not uri.startswith(("s3://", "gs://", "az://", "http://", "https://")):
            import shutil
            from pathlib import Path

            path = Path(uri)
            if path.exists():
                shutil.rmtree(path)

    def delete(
        self,
        conn: SQLBackend,  # noqa: ARG002
        storage_config: StorageConfig,
        key: FeatureKey,
        filters: Sequence[nw.Expr] | None,  # noqa: ARG002
        *,
        with_feature_history: bool,  # noqa: ARG002
    ) -> None:
        raise NotImplementedError("Row-level deletion not yet supported for DuckDB Lance handler.")
