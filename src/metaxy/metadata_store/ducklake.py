"""DuckLake metadata store backed by DuckDB with typed configuration models.

The store reuses the DuckDB native Narwhals components. All that differs is how
the DuckDB connection is prepared: DuckLake plugins are installed, secrets are
created, and the DuckLake catalog is attached/selected.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from metaxy.metadata_store.duckdb import DuckDBMetadataStore, ExtensionSpec

try:  # pragma: no cover - optional dependency in some environments
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    _CATALOG_EXCEPTIONS: tuple[type[Exception], ...] = ()
else:  # pragma: no cover - runtime dependent
    candidates: list[type[Exception]] = []
    for name in ("CatalogException", "CatalogError"):
        exc = getattr(duckdb, name, None)
        if isinstance(exc, type) and issubclass(exc, Exception):
            candidates.append(exc)
    if not candidates and hasattr(duckdb, "Error"):
        base_exc = getattr(duckdb, "Error")
        if isinstance(base_exc, type) and issubclass(base_exc, Exception):
            candidates.append(base_exc)
    _CATALOG_EXCEPTIONS = tuple(candidates)


# ------------------------------------------------------------------------------
# Metadata backend configuration


class DuckLakeMetadataBackend(BaseModel):
    """Base class for typed DuckLake metadata backend configuration."""

    type: str

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]:
        """Return SQL fragments required for ATTACH metadata configuration."""
        raise NotImplementedError


class DuckLakePostgresMetadataBackend(DuckLakeMetadataBackend):
    """Postgres catalog backend for DuckLake."""

    type: Literal["postgres"] = "postgres"
    host: str = Field(
        default_factory=lambda: os.getenv("DUCKLAKE_PG_HOST", "localhost")
    )
    port: int = Field(default=5432)
    database: str
    user: str
    password: str

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]:
        secret_name = f"secret_catalog_{alias}"
        secret_sql = (
            f"CREATE OR REPLACE SECRET {secret_name} ("
            f" TYPE postgres, HOST '{self.host}', PORT {self.port},"
            f" DATABASE '{self.database}', USER '{self.user}',"
            f" PASSWORD '{self.password}'"
            " );"
        )
        metadata_params = (
            "METADATA_PATH '', "
            f"METADATA_PARAMETERS MAP {{'TYPE': 'postgres', 'SECRET': '{secret_name}'}}"
        )
        return secret_sql, metadata_params


class DuckLakeSqliteMetadataBackend(DuckLakeMetadataBackend):
    """SQLite catalog backend for DuckLake."""

    type: Literal["sqlite"] = "sqlite"
    path: str = Field(description="Path to the SQLite database file.")

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH '{self.path}'"


class DuckLakeDuckDBMetadataBackend(DuckLakeMetadataBackend):
    """DuckDB catalog backend for DuckLake."""

    type: Literal["duckdb"] = "duckdb"
    path: str = Field(description="Path to the DuckDB database file.")

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH '{self.path}'"


DuckLakeMetadataBackendConfig = (
    DuckLakePostgresMetadataBackend
    | DuckLakeSqliteMetadataBackend
    | DuckLakeDuckDBMetadataBackend
)


# ------------------------------------------------------------------------------
# Storage backend configuration


class DuckLakeStorageBackend(BaseModel):
    """Base class for typed DuckLake storage backend configuration."""

    type: str

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]:
        """Return SQL fragments required for ATTACH storage configuration."""
        raise NotImplementedError


class DuckLakeS3StorageBackend(DuckLakeStorageBackend):
    """S3-compatible object storage configuration."""

    type: Literal["s3"] = "s3"
    endpoint_url: str
    bucket: str
    prefix: str | None = None
    aws_access_key_id: str
    aws_secret_access_key: str
    region: str = "us-east-1"
    use_ssl: bool = True
    url_style: str = Field(
        default="path",
        description="URL style for S3 ('path' or 'virtual').",
    )

    @property
    def full_data_path(self) -> str:
        base = f"s3://{self.bucket}"
        if self.prefix:
            clean_prefix = self.prefix.strip("/")
            if clean_prefix:
                return f"{base}/{clean_prefix}/"
        return f"{base}/"

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]:
        secret_name = f"secret_storage_{alias}"
        secret_sql = (
            f"CREATE OR REPLACE SECRET {secret_name} ("
            " TYPE S3,"
            f" KEY_ID '{self.aws_access_key_id}',"
            f" SECRET '{self.aws_secret_access_key}',"
            f" ENDPOINT '{self.endpoint_url}',"
            f" URL_STYLE '{self.url_style}',"
            f" REGION '{self.region}',"
            f" USE_SSL {'true' if self.use_ssl else 'false'},"
            f" SCOPE 's3://{self.bucket}'"
            " );"
        )
        data_path_sql = f"DATA_PATH '{self.full_data_path}'"
        return secret_sql, data_path_sql


class DuckLakeLocalStorageBackend(DuckLakeStorageBackend):
    """Local filesystem storage configuration."""

    type: Literal["local"] = "local"
    path: str

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]:
        return "", f"DATA_PATH '{self.path}'"


DuckLakeStorageBackendConfig = DuckLakeS3StorageBackend | DuckLakeLocalStorageBackend


# ------------------------------------------------------------------------------
# Helpers / attachment manager


_METADATA_ADAPTER = TypeAdapter(DuckLakeMetadataBackendConfig)
_STORAGE_ADAPTER = TypeAdapter(DuckLakeStorageBackendConfig)


def _format_attach_options(options: Mapping[str, Any] | None) -> str:
    if not options:
        return ""

    def _format_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    parts: list[str] = []
    for key, value in sorted(options.items()):
        formatted = _format_value(value)
        if formatted is None:
            continue
        parts.append(f"{str(key).upper()} {formatted}")

    return f" ({', '.join(parts)})" if parts else ""


@dataclass
class DuckLakeAttachmentConfig:
    """Configuration payload used to attach DuckLake to a DuckDB connection."""

    metadata_backend: DuckLakeMetadataBackendConfig
    storage_backend: DuckLakeStorageBackendConfig
    alias: str = "ducklake"
    plugins: Sequence[str] = ("ducklake",)
    attach_options: Mapping[str, Any] | None = None


class _PreviewCursor:
    """Collects commands for previewing DuckLake attachment SQL."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, command: str) -> None:
        self.commands.append(command.strip())

    def close(self) -> None:  # pragma: no cover - no-op in preview mode
        pass


class _PreviewConnection:
    """Mock DuckDB connection used for previewing generated SQL."""

    def __init__(self) -> None:
        self._cursor = _PreviewCursor()

    def cursor(self) -> _PreviewCursor:
        return self._cursor


class DuckLakeAttachmentManager:
    """Responsible for configuring a DuckDB connection for DuckLake usage."""

    def __init__(self, config: DuckLakeAttachmentConfig):
        self._config = config

    def configure(self, conn) -> None:
        cursor = conn.cursor()
        try:
            for plugin in self._config.plugins:
                cursor.execute(f"INSTALL {plugin};")
                cursor.execute(f"LOAD {plugin};")

            if _CATALOG_EXCEPTIONS:
                try:
                    cursor.execute(f"DETACH {self._config.alias};")
                except _CATALOG_EXCEPTIONS:  # type: ignore[arg-type]
                    pass
            else:
                cursor.execute(f"DETACH {self._config.alias};")

            metadata_secret_sql, metadata_params_sql = (
                self._config.metadata_backend.get_ducklake_sql_parts(self._config.alias)
            )
            storage_secret_sql, storage_params_sql = (
                self._config.storage_backend.get_ducklake_sql_parts(self._config.alias)
            )

            if metadata_secret_sql:
                cursor.execute(metadata_secret_sql)
            if storage_secret_sql:
                cursor.execute(storage_secret_sql)

            ducklake_secret = f"secret_{self._config.alias}"
            cursor.execute(
                f"CREATE OR REPLACE SECRET {ducklake_secret} ("
                " TYPE DUCKLAKE,"
                f" {metadata_params_sql},"
                f" {storage_params_sql}"
                " );"
            )

            options_clause = _format_attach_options(self._config.attach_options)
            cursor.execute(
                f"ATTACH 'ducklake:{ducklake_secret}' AS {self._config.alias}{options_clause};"
            )
            cursor.execute(f"USE {self._config.alias};")
        finally:
            cursor.close()

    def preview_sql(self) -> list[str]:
        """Return the SQL statements that would be executed during configure()."""
        preview_conn = _PreviewConnection()
        self.configure(preview_conn)
        return preview_conn.cursor().commands


# ------------------------------------------------------------------------------
# Public store


class DuckLakeMetadataStore(DuckDBMetadataStore):
    """DuckLake metadata store that reuses DuckDB native components."""

    def __init__(
        self,
        *,
        metadata_backend: DuckLakeMetadataBackendConfig | dict[str, Any],
        storage_backend: DuckLakeStorageBackendConfig | dict[str, Any],
        alias: str = "ducklake",
        plugins: Iterable[str] | None = None,
        attach_options: Mapping[str, Any] | None = None,
        extensions: Sequence[ExtensionSpec | str] | None = None,
        database: str = ":memory:",
        config: Mapping[str, str] | None = None,
        **kwargs: Any,
    ):
        metadata_cfg = (
            metadata_backend
            if isinstance(metadata_backend, DuckLakeMetadataBackend)
            else _METADATA_ADAPTER.validate_python(metadata_backend)
        )
        storage_cfg = (
            storage_backend
            if isinstance(storage_backend, DuckLakeStorageBackend)
            else _STORAGE_ADAPTER.validate_python(storage_backend)
        )

        plugin_list = list(plugins or ("ducklake",))

        base_extensions = list(extensions or [])
        extension_names = {
            ext if isinstance(ext, str) else ext.get("name", "")
            for ext in base_extensions
        }
        for plugin in plugin_list:
            if plugin not in extension_names:
                base_extensions.append(plugin)
                extension_names.add(plugin)

        super().__init__(
            database=database,
            config=dict(config or {}),
            extensions=base_extensions,
            **kwargs,
        )

        self.metadata_backend_config = metadata_cfg
        self.storage_backend_config = storage_cfg
        self._ducklake_alias = alias
        self._ducklake_attachment = DuckLakeAttachmentManager(
            DuckLakeAttachmentConfig(
                metadata_backend=metadata_cfg,
                storage_backend=storage_cfg,
                alias=alias,
                plugins=plugin_list,
                attach_options=dict(attach_options or {}),
            )
        )

    def open(self) -> None:
        """Open DuckDB connection and attach DuckLake catalog."""
        super().open()
        if self._conn is None:
            raise RuntimeError("DuckLakeMetadataStore failed to open DuckDB connection")
        self._ducklake_attachment.configure(self._conn)

    def get_attachment_manager(self) -> DuckLakeAttachmentManager:
        """Expose the configured attachment manager for advanced scenarios."""
        return self._ducklake_attachment

    def preview_attachment_sql(self) -> list[str]:
        """Return SQL statements executed when attaching DuckLake."""
        return self._ducklake_attachment.preview_sql()
