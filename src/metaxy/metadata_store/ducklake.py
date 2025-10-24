"""DuckLake metadata store backed by DuckDB.

This store reuses the DuckDB native components (Narwhals joiner/calculator/diff)
and only customises how the DuckDB connection is initialised so that the
DuckLake catalog is attached and selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from metaxy.metadata_store.duckdb import DuckDBMetadataStore, ExtensionSpec


def _build_metadata_backend_parts(
    config: Mapping[str, Any], alias: str
) -> tuple[str, str]:
    backend_type = str(config.get("type", "")).lower()

    if backend_type == "postgres":
        host = config.get("host", "localhost")
        port = int(config.get("port", 5432))
        database = config["database"]
        user = config["user"]
        password = config["password"]

        secret_name = f"secret_catalog_{alias}"
        secret_sql = (
            f"CREATE OR REPLACE SECRET {secret_name} ("
            f" TYPE postgres, HOST '{host}', PORT {port},"
            f" DATABASE '{database}', USER '{user}',"
            f" PASSWORD '{password}'"
            " );"
        )
        metadata_params = (
            "METADATA_PATH '', "
            f"METADATA_PARAMETERS MAP {{'TYPE': 'postgres', 'SECRET': '{secret_name}'}}"
        )
        return secret_sql, metadata_params

    if backend_type == "sqlite":
        path = config["path"]
        return "", f"METADATA_PATH '{path}'"

    if backend_type == "duckdb":
        path = config["path"]
        return "", f"METADATA_PATH '{path}'"

    raise ValueError(f"Unsupported DuckLake metadata backend: {backend_type}")


def _build_storage_backend_parts(
    config: Mapping[str, Any], alias: str
) -> tuple[str, str]:
    storage_type = str(config.get("type", "")).lower()

    if storage_type == "s3":
        endpoint = config["endpoint_url"]
        bucket = config["bucket"]
        prefix = config.get("prefix")
        access_key = config["aws_access_key_id"]
        secret_key = config["aws_secret_access_key"]
        region = config.get("region", "us-east-1")
        use_ssl = bool(config.get("use_ssl", True))
        url_style = config.get("url_style", "path")

        secret_name = f"secret_storage_{alias}"
        secret_sql = (
            f"CREATE OR REPLACE SECRET {secret_name} ("
            " TYPE S3,"
            f" KEY_ID '{access_key}',"
            f" SECRET '{secret_key}',"
            f" ENDPOINT '{endpoint}',"
            f" URL_STYLE '{url_style}',"
            f" REGION '{region}',"
            f" USE_SSL {'true' if use_ssl else 'false'},"
            f" SCOPE 's3://{bucket}'"
            " );"
        )

        clean_prefix = ""
        if isinstance(prefix, str) and prefix.strip():
            clean_prefix = prefix.strip("/ ")
            clean_prefix = f"{clean_prefix}/"

        data_path = f"s3://{bucket}/{clean_prefix}" if clean_prefix else f"s3://{bucket}/"
        return secret_sql, f"DATA_PATH '{data_path}'"

    if storage_type == "local":
        path = config["path"]
        return "", f"DATA_PATH '{path}'"

    raise ValueError(f"Unsupported DuckLake storage backend: {storage_type}")


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

    if not parts:
        return ""

    return f" ({', '.join(parts)})"


@dataclass
class DuckLakeAttachmentConfig:
    """Configuration used to attach DuckLake to a DuckDB connection."""

    metadata_backend: Mapping[str, Any]
    storage_backend: Mapping[str, Any]
    alias: str = "ducklake"
    plugins: Sequence[str] = ("ducklake",)
    attach_options: Mapping[str, Any] | None = None


class DuckLakeAttachmentManager:
    """Helper that installs plugins and issues ATTACH for DuckLake."""

    def __init__(self, config: DuckLakeAttachmentConfig):
        self._config = config

    def configure(self, conn) -> None:
        cursor = conn.cursor()
        try:
            for plugin in self._config.plugins:
                cursor.execute(f"INSTALL {plugin};")
                cursor.execute(f"LOAD {plugin};")

            try:
                cursor.execute(f"DETACH {self._config.alias};")
            except Exception:
                pass

            metadata_secret_sql, metadata_params_sql = _build_metadata_backend_parts(
                self._config.metadata_backend, self._config.alias
            )
            storage_secret_sql, storage_params_sql = _build_storage_backend_parts(
                self._config.storage_backend, self._config.alias
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


class DuckLakeMetadataStore(DuckDBMetadataStore):
    """DuckLake metadata store that reuses DuckDB native components."""

    def __init__(
        self,
        *,
        metadata_backend: Mapping[str, Any],
        storage_backend: Mapping[str, Any],
        alias: str = "ducklake",
        plugins: Iterable[str] | None = None,
        attach_options: Mapping[str, Any] | None = None,
        extensions: Sequence[ExtensionSpec | str] | None = None,
        database: str = ":memory:",
        config: Mapping[str, str] | None = None,
        **kwargs: Any,
    ):
        plugin_list = list(plugins or ("ducklake",))

        base_extensions = list(extensions or [])
        extension_names = {
            ext if isinstance(ext, str) else ext.get("name", "") for ext in base_extensions
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

        self.metadata_backend_config = dict(metadata_backend)
        self.storage_backend_config = dict(storage_backend)
        self._ducklake_attachment = DuckLakeAttachmentManager(
            DuckLakeAttachmentConfig(
                metadata_backend=self.metadata_backend_config,
                storage_backend=self.storage_backend_config,
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
