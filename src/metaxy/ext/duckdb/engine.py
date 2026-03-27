"""DuckDB compute engine using Ibis."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from duckdb import DuckDBPyConnection  # noqa: TID252

from metaxy.ext.duckdb.config import (
    DuckDBMetadataStoreConfig,
    ExtensionSpec,
    _normalise_extensions,
)
from metaxy.metadata_store.ibis_compute_engine import IbisComputeEngine
from metaxy.metadata_store.storage_handler import StorageHandler
from metaxy.metadata_store.types import AccessMode
from metaxy.versioning.types import HashAlgorithm


class DuckDBEngine(IbisComputeEngine):
    """Compute engine for DuckDB backends using Ibis."""

    def __init__(
        self,
        database: str | Path,
        *,
        config: dict[str, str] | None = None,
        extensions: Sequence[str | ExtensionSpec] | None = None,
        auto_create_tables: bool = False,
        handler: StorageHandler | None = None,
    ) -> None:
        self.database = str(database)
        self.extensions: list[ExtensionSpec] = _normalise_extensions(extensions or [])

        if "hashfuncs" not in {ext.name for ext in self.extensions}:
            self.extensions.append(ExtensionSpec(name="hashfuncs", repository="community"))

        connection_params: dict[str, Any] = {"database": self.database}
        if config:
            connection_params.update(config)

        super().__init__(
            backend="duckdb",
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handler=handler,
        )

    def open(self, mode: AccessMode) -> None:
        if mode == "r":
            db = self.connection_params.get("database", "")
            db = str(db) if db is not None else ""
            is_in_memory = db in {"", ":memory:"}
            scheme = urlsplit(db).scheme if db else ""
            is_windows_drive_path = len(scheme) == 1 and bool(Path(db).drive)
            is_local_file = bool(db) and not is_in_memory and (scheme == "" or is_windows_drive_path)
            is_remote = not (is_in_memory or is_local_file)
            if is_remote or (is_local_file and Path(db).exists()):
                self.connection_params["read_only"] = True
            else:
                self.connection_params.pop("read_only", None)
        else:
            self.connection_params.pop("read_only", None)

        super().open(mode)
        self._load_extensions()

    def close(self) -> None:
        super().close()

    def _create_hash_functions(self) -> dict:
        import ibis

        hash_functions = {}

        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:  # ty: ignore[empty-body]  # noqa: N802
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:  # ty: ignore[empty-body]  # noqa: N802
            ...

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str:  # ty: ignore[empty-body]  # noqa: N802
            ...

        def md5_hash(col_expr):  # noqa: ANN001, ANN202
            return LOWER(MD5(col_expr.cast(str)))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        if "hashfuncs" in {ext.name for ext in self.extensions}:

            @ibis.udf.scalar.builtin
            def xxh32(x: str) -> int:  # ty: ignore[empty-body]
                ...

            @ibis.udf.scalar.builtin
            def xxh64(x: str) -> int:  # ty: ignore[empty-body]
                ...

            def xxhash32_hash(col_expr):  # noqa: ANN001, ANN202
                return xxh32(col_expr.cast(str)).cast(str)

            def xxhash64_hash(col_expr):  # noqa: ANN001, ANN202
                return xxh64(col_expr.cast(str)).cast(str)

            hash_functions[HashAlgorithm.XXHASH32] = xxhash32_hash
            hash_functions[HashAlgorithm.XXHASH64] = xxhash64_hash

        return hash_functions

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    def _duckdb_raw_connection(self) -> DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDB connection is not open.")

        candidate = self._conn.con  # ty: ignore[unresolved-attribute]

        if not isinstance(candidate, DuckDBPyConnection):
            raise TypeError(f"Expected DuckDB backend 'con' to be DuckDBPyConnection, got {type(candidate).__name__}")

        return candidate

    def _load_extensions(self) -> None:
        if not self.extensions:
            return

        duckdb_conn = self._duckdb_raw_connection()
        for ext in self.extensions:
            duckdb_conn.install_extension(ext.name, repository=ext.repository)
            duckdb_conn.load_extension(ext.name)
            for sql in ext.init_sql:
                duckdb_conn.execute(sql)

    @property
    def sqlalchemy_url(self) -> str:
        return f"duckdb:///{self.database}"

    def display(self) -> str:
        from metaxy.metadata_store.utils import sanitize_uri

        return f"DuckDBEngine(database={sanitize_uri(self.database)})"

    @classmethod
    def config_model(cls) -> type[DuckDBMetadataStoreConfig]:
        return DuckDBMetadataStoreConfig
