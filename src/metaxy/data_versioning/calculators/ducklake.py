"""DuckLake-specific data version calculator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from metaxy.data_versioning.calculators.duckdb import (
    DuckDBDataVersionCalculator,
    ExtensionSpec,
)
from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    import ibis


class DuckLakeDataVersionCalculator(DuckDBDataVersionCalculator):
    """DuckLake calculator that ensures DuckLake plugins are installed."""

    def __init__(
        self,
        hash_sql_generators: dict[HashAlgorithm, HashSQLGenerator],
        *,
        alias: str = "ducklake",
        extensions: Iterable[ExtensionSpec | str] | None = None,
        connection: "ibis.BaseBackend | None" = None,
    ):
        ext_list = list(extensions or [])
        extension_names = {
            ext if isinstance(ext, str) else ext.get("name", "") for ext in ext_list
        }
        if "ducklake" not in extension_names:
            ext_list.append("ducklake")
            extension_names.add("ducklake")
        if "hashfuncs" not in extension_names:
            ext_list.append("hashfuncs")

        super().__init__(
            hash_sql_generators=hash_sql_generators,
            extensions=ext_list,
            connection=connection,
        )
        self.alias = alias

    def set_connection(self, connection: "ibis.BaseBackend") -> None:
        super().set_connection(connection)
        # Ensure all queries operate within the DuckLake catalog
        try:
            connection.raw_sql(f"USE {self.alias}")  # type: ignore[attr-defined]
        except Exception:
            # Falling back to pre-configured catalog if USE fails (e.g., already set)
            pass
