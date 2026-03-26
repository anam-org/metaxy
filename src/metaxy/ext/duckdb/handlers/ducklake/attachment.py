"""DuckLake attachment manager for DuckDB connections."""

from __future__ import annotations

from duckdb import DuckDBPyConnection  # noqa: TID252

from metaxy.ext.duckdb.handlers.ducklake.config import (
    DuckDBCatalogConfig,
    DuckLakeConfig,
    MotherDuckCatalogConfig,
    PostgresCatalogConfig,
    SQLiteCatalogConfig,
    format_attach_options,
)


class DuckLakeAttachmentManager:
    """Responsible for configuring a DuckDB connection for DuckLake usage."""

    def __init__(self, config: DuckLakeConfig, *, store_name: str | None = None, use_catalog: bool = True):
        self._config = config
        self._secret_suffix = f"{store_name}_{config.alias}" if store_name else config.alias
        self._use_catalog = use_catalog
        self._attached: bool = False

    def _build_sql_statements(self) -> list[str]:
        """Build the full list of SQL statements for DuckLake attachment (secrets + ATTACH)."""
        statements: list[str] = []

        if isinstance(self._config.catalog, MotherDuckCatalogConfig):
            return self._build_motherduck_statements(statements)

        return self._build_standard_statements(statements)

    def _build_motherduck_statements(self, statements: list[str]) -> list[str]:
        """Build SQL statements for MotherDuck-managed DuckLake.

        Fully managed MotherDuck DuckLake databases are available to any
        MotherDuck connection via ``USE``.  When a ``storage`` is
        provided (BYOB mode), the DuckLake database is created with a custom
        ``DATA_PATH`` and storage secrets are created ``IN MOTHERDUCK``.
        """
        assert isinstance(self._config.catalog, MotherDuckCatalogConfig)
        md = self._config.catalog
        if md.region is not None:
            statements.append(f"SET s3_region='{md.region}';")
        if self._config.storage is not None:
            storage_secret_sql, data_path_sql = self._config.storage.sql_parts(
                self._secret_suffix, secret_storage="MOTHERDUCK"
            )
            statements.append(f"CREATE DATABASE IF NOT EXISTS {md.database} (TYPE DUCKLAKE, {data_path_sql});")
            if storage_secret_sql:
                statements.append(storage_secret_sql)
        if self._use_catalog:
            statements.append(f"USE {md.database};")
        return statements

    def _build_standard_statements(self, statements: list[str]) -> list[str]:
        """Build SQL statements for self-managed DuckLake backends."""
        catalog = self._config.catalog
        assert isinstance(catalog, DuckDBCatalogConfig | SQLiteCatalogConfig | PostgresCatalogConfig)
        assert self._config.storage is not None
        metadata_secret_sql, metadata_params_sql = catalog.sql_parts(self._secret_suffix)
        storage_secret_sql, storage_params_sql = self._config.storage.sql_parts(self._secret_suffix)

        if metadata_secret_sql:
            statements.append(metadata_secret_sql)
        if storage_secret_sql:
            statements.append(storage_secret_sql)

        ducklake_secret = f"metaxy_generated_{self._secret_suffix}"
        statements.append(
            f"CREATE OR REPLACE SECRET {ducklake_secret} ("
            " TYPE DUCKLAKE,"
            f" {metadata_params_sql},"
            f" {storage_params_sql}"
            " );"
        )

        options = dict(self._config.attach_options)
        if self._config.data_inlining_row_limit is not None:
            options["data_inlining_row_limit"] = self._config.data_inlining_row_limit
        options_clause = format_attach_options(options)
        statements.append(f"ATTACH IF NOT EXISTS 'ducklake:{ducklake_secret}' AS {self._config.alias}{options_clause};")
        if self._use_catalog:
            statements.append(f"USE {self._config.alias};")
        return statements

    def configure(self, conn: DuckDBPyConnection) -> None:
        """Execute DuckLake attachment statements on a live DuckDB connection."""
        if self._attached:
            return

        for stmt in self._build_sql_statements():
            conn.execute(stmt)
        self._attached = True

    def preview_sql(self) -> list[str]:
        """Return the SQL statements that would be executed during configure()."""
        return self._build_sql_statements()
