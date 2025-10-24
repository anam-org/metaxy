"""DuckLake metadata store built on top of the DuckDB backend.

This module configures an in-memory DuckDB connection, installs DuckLake
and related plugins, and attaches a DuckLake instance using secrets that
reference configurable metadata and storage backends.
"""

from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Literal

from metaxy.metadata_store.duckdb import DuckDBMetadataStore, ExtensionSpec


class DuckLakeMetadataBackend(BaseModel):
    """Base class for DuckLake metadata backend configuration."""

    type: str

    def build_sql_parts(self, alias: str) -> tuple[str, str]:
        """Return SQL fragments for secrets/parameters used during ATTACH."""
        raise NotImplementedError


class DuckLakePostgresMetadataBackend(DuckLakeMetadataBackend):
    """Postgres catalog backend for DuckLake."""

    type: Literal["postgres"] = "postgres"
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str
    user: str
    password: str

    def build_sql_parts(self, alias: str) -> tuple[str, str]:
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

    def build_sql_parts(self, alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH '{self.path}'"


class DuckLakeDuckDBMetadataBackend(DuckLakeMetadataBackend):
    """DuckDB catalog backend for DuckLake."""

    type: Literal["duckdb"] = "duckdb"
    path: str = Field(description="Path to the DuckDB database file.")

    def build_sql_parts(self, alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH '{self.path}'"


class DuckLakeStorageBackend(BaseModel):
    """Base class for DuckLake storage backend configuration."""

    type: str

    def build_sql_parts(self, alias: str) -> tuple[str, str]:
        """Return SQL fragments for secrets/parameters used during ATTACH."""
        raise NotImplementedError


class DuckLakeS3StorageBackend(DuckLakeStorageBackend):
    """S3-compatible object storage for DuckLake."""

    type: Literal["s3"] = "s3"
    endpoint_url: str
    bucket: str
    prefix: str | None = None
    aws_access_key_id: str
    aws_secret_access_key: str
    region: str = "us-east-1"
    use_ssl: bool = True
    url_style: Literal["path", "virtual"] = "path"

    @property
    def full_data_path(self) -> str:
        base = f"s3://{self.bucket}"
        if self.prefix:
            clean_prefix = self.prefix.strip("/")
            return f"{base}/{clean_prefix}/"
        return f"{base}/"

    def build_sql_parts(self, alias: str) -> tuple[str, str]:
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
    """Local filesystem storage backend for DuckLake."""

    type: Literal["local"] = "local"
    path: str

    def build_sql_parts(self, alias: str) -> tuple[str, str]:
        return "", f"DATA_PATH '{self.path}'"


DuckLakeMetadataBackendConfig = (
    DuckLakePostgresMetadataBackend
    | DuckLakeSqliteMetadataBackend
    | DuckLakeDuckDBMetadataBackend
)
DuckLakeStorageBackendConfig = DuckLakeS3StorageBackend | DuckLakeLocalStorageBackend

_METADATA_ADAPTER = TypeAdapter(DuckLakeMetadataBackendConfig)
_STORAGE_ADAPTER = TypeAdapter(DuckLakeStorageBackendConfig)


@dataclass
class DuckLakeAttachmentConfig:
    """Configuration payload used to attach DuckLake to DuckDB."""

    metadata_backend: DuckLakeMetadataBackendConfig
    storage_backend: DuckLakeStorageBackendConfig
    alias: str = "ducklake"
    plugins: Iterable[str] = ("ducklake",)
    attach_options: dict[str, Any] | None = None


class DuckLakeAttachmentManager:
    """Responsible for configuring a DuckDB connection for DuckLake usage."""

    def __init__(self, config: DuckLakeAttachmentConfig):
        self._config = config

    def _build_attach_options_clause(self) -> str:
        options = self._config.attach_options or {}
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

        if not parts:
            return ""

        joined = ", ".join(parts)
        return f" ({joined})"

    def configure(self, conn) -> None:
        """Install plugins, create secrets, and attach DuckLake."""
        cursor = conn.cursor()
        try:
            for plugin in self._config.plugins:
                cursor.execute(f"INSTALL {plugin};")
                cursor.execute(f"LOAD {plugin};")

            # Attempt to detach existing attachment to allow reconfiguration
            try:
                cursor.execute(f"DETACH {self._config.alias};")
            except Exception:
                pass

            metadata_secret_sql, metadata_params_sql = (
                self._config.metadata_backend.build_sql_parts(self._config.alias)
            )
            storage_secret_sql, storage_params_sql = (
                self._config.storage_backend.build_sql_parts(self._config.alias)
            )

            if metadata_secret_sql:
                cursor.execute(metadata_secret_sql)
            if storage_secret_sql:
                cursor.execute(storage_secret_sql)

            ducklake_secret = f"secret_{self._config.alias}"
            cursor.execute(
                "CREATE OR REPLACE SECRET {name} ("
                " TYPE DUCKLAKE,"
                " {metadata_params},"
                " {storage_params}"
                " );".format(
                    name=ducklake_secret,
                    metadata_params=metadata_params_sql,
                    storage_params=storage_params_sql,
                )
            )

            options_clause = self._build_attach_options_clause()
            cursor.execute(
                f"ATTACH 'ducklake:{ducklake_secret}' AS {self._config.alias}{options_clause};"
            )
            cursor.execute(f"USE {self._config.alias};")
        finally:
            cursor.close()


class DuckLakeMetadataStore(DuckDBMetadataStore):
    """Metadata store that connects to DuckLake via DuckDB attachments."""

    def __init__(
        self,
        *,
        metadata_backend: DuckLakeMetadataBackendConfig | dict[str, Any],
        storage_backend: DuckLakeStorageBackendConfig | dict[str, Any],
        alias: str = "ducklake",
        plugins: Iterable[str] | None = None,
        attach_options: dict[str, Any] | None = None,
        extensions: list[ExtensionSpec | str] | None = None,
        database: str = ":memory:",
        config: dict[str, str] | None = None,
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

        base_extensions: list[ExtensionSpec | str] = list(extensions or [])
        plugin_list = list(plugins or ["ducklake"])

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
            config=config,
            extensions=base_extensions,
            **kwargs,
        )

        self.metadata_backend_config = metadata_cfg
        self.storage_backend_config = storage_cfg
        self._ducklake_alias = alias
        self._ducklake_plugins = plugin_list
        self._ducklake_attach_options = attach_options or {}

        self._ducklake_attachment = DuckLakeAttachmentManager(
            DuckLakeAttachmentConfig(
                metadata_backend=metadata_cfg,
                storage_backend=storage_cfg,
                alias=alias,
                plugins=plugin_list,
                attach_options=self._ducklake_attach_options,
            )
        )

    def open(self) -> None:
        """Open DuckDB connection and attach DuckLake."""
        super().open()
        if self._conn is None:
            raise RuntimeError("DuckLakeMetadataStore failed to open DuckDB connection")
        self._ducklake_attachment.configure(self._conn)

    def _create_native_components(self):
        """Create DuckLake-specific native components."""
        from metaxy.data_versioning.calculators.base import DataVersionCalculator
        from metaxy.data_versioning.diff.base import MetadataDiffResolver
        from metaxy.data_versioning.joiners.base import UpstreamJoiner
        from metaxy.data_versioning.calculators.ducklake import (
            DuckLakeDataVersionCalculator,
        )
        from metaxy.data_versioning.diff.ducklake import DuckLakeDiffResolver
        from metaxy.data_versioning.joiners.ducklake import DuckLakeJoiner

        if self._conn is None:
            raise RuntimeError(
                "Cannot create native components: store is not open. "
                "Ensure store is used as context manager."
            )

        import ibis.expr.types as ir

        joiner: UpstreamJoiner[ir.Table] = DuckLakeJoiner(
            backend=self._conn, alias=self._ducklake_alias
        )
        calculator: DataVersionCalculator[ir.Table] = DuckLakeDataVersionCalculator(
            hash_sql_generators=self.hash_sql_generators,
            alias=self._ducklake_alias,
            extensions=self.extensions,
            connection=self._conn,
        )

        diff_resolver: MetadataDiffResolver[ir.Table] = DuckLakeDiffResolver(
            backend=self._conn, alias=self._ducklake_alias
        )

        return joiner, calculator, diff_resolver
