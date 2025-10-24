"""DuckLake metadata store backed by DuckDB.

The store reuses the DuckDB native Narwhals components. All that differs is how
the DuckDB connection is prepared: DuckLake plugins are installed, secrets are
created, and the DuckLake catalog is attached/selected.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from metaxy.metadata_store.duckdb import DuckDBMetadataStore, ExtensionSpec

try:  # pragma: no cover - optional dependency in some environments
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]
    _CATALOG_EXCEPTIONS: tuple[type[Exception], ...] = ()
    DuckDBConnection = Any  # type: ignore[assignment]
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
    DuckDBConnection = duckdb.DuckDBPyConnection  # type: ignore[attr-defined]


@runtime_checkable
class SupportsDuckLakeParts(Protocol):
    """Protocol for objects that can produce DuckLake attachment SQL fragments."""

    def get_ducklake_sql_parts(self, alias: str) -> tuple[str, str]: ...


@runtime_checkable
class SupportsModelDump(Protocol):
    """Protocol for Pydantic-like objects that expose a model_dump method."""

    def model_dump(self) -> Mapping[str, Any]: ...


DuckLakeBackendInput = Mapping[str, Any] | SupportsDuckLakeParts | SupportsModelDump
DuckLakeBackend = SupportsDuckLakeParts | dict[str, Any]


def _coerce_backend_config(
    backend: DuckLakeBackendInput, *, role: str
) -> DuckLakeBackend:
    if isinstance(backend, SupportsDuckLakeParts):
        return backend
    if isinstance(backend, SupportsModelDump):
        return dict(backend.model_dump())
    if isinstance(backend, Mapping):
        return dict(backend)
    raise TypeError(
        f"DuckLake {role} must be a mapping or expose get_ducklake_sql_parts()/model_dump(), "
        f"got {type(backend)!r}."
    )


def _resolve_metadata_backend(backend: DuckLakeBackend, alias: str) -> tuple[str, str]:
    if isinstance(backend, SupportsDuckLakeParts):
        return backend.get_ducklake_sql_parts(alias)
    return _metadata_sql_from_mapping(backend, alias)


def _resolve_storage_backend(backend: DuckLakeBackend, alias: str) -> tuple[str, str]:
    if isinstance(backend, SupportsDuckLakeParts):
        return backend.get_ducklake_sql_parts(alias)
    return _storage_sql_from_mapping(backend, alias)


def _metadata_sql_from_mapping(
    config: Mapping[str, Any], alias: str
) -> tuple[str, str]:
    backend_type = str(config.get("type", "")).lower()
    if backend_type == "postgres":
        return _metadata_postgres_sql(config, alias)
    if backend_type in {"sqlite", "duckdb"}:
        path = config.get("path")
        if not path:
            raise ValueError(
                "DuckLake metadata backend of type "
                f"'{backend_type}' requires a 'path' entry."
            )
        literal_path = _stringify_scalar(path)
        return "", f"METADATA_PATH {literal_path}"
    raise ValueError(f"Unsupported DuckLake metadata backend type: {backend_type!r}")


def _metadata_postgres_sql(config: Mapping[str, Any], alias: str) -> tuple[str, str]:
    database = config.get("database")
    user = config.get("user")
    password = config.get("password")
    if database is None or user is None or password is None:
        raise ValueError(
            "DuckLake postgres metadata backend requires 'database', 'user', and 'password'."
        )
    host = config.get("host") or os.getenv("DUCKLAKE_PG_HOST", "localhost")
    port_value = config.get("port", 5432)
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DuckLake postgres metadata backend requires 'port' to be an integer."
        ) from exc
    secret_params: dict[str, Any] = {
        "HOST": host,
        "PORT": port,
        "DATABASE": database,
        "USER": user,
        "PASSWORD": password,
    }

    extra_params = config.get("secret_parameters")
    if isinstance(extra_params, Mapping):
        for key, value in extra_params.items():
            secret_params[str(key).upper()] = value

    secret_name = f"secret_catalog_{alias}"
    secret_sql = _build_secret_sql(secret_name, "postgres", secret_params)
    metadata_params = (
        "METADATA_PATH '', "
        f"METADATA_PARAMETERS MAP {{'TYPE': 'postgres', 'SECRET': '{secret_name}'}}"
    )
    return secret_sql, metadata_params


def _storage_sql_from_mapping(config: Mapping[str, Any], alias: str) -> tuple[str, str]:
    storage_type = str(config.get("type", "")).lower()
    if storage_type == "s3":
        return _storage_s3_sql(config, alias)
    if storage_type == "local":
        path = config.get("path")
        if not path:
            raise ValueError("DuckLake local storage backend requires 'path'.")
        literal_path = _stringify_scalar(path)
        return "", f"DATA_PATH {literal_path}"
    raise ValueError(f"Unsupported DuckLake storage backend type: {storage_type!r}")


def _storage_s3_sql(config: Mapping[str, Any], alias: str) -> tuple[str, str]:
    secret_name = f"secret_storage_{alias}"
    secret_config = config.get("secret")
    secret_params: dict[str, Any]
    if isinstance(secret_config, Mapping):
        secret_params = {str(k): v for k, v in secret_config.items()}
    else:  # Backward-compatible typed configuration
        required_keys = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "endpoint_url",
            "bucket",
        ]
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(
                "DuckLake S3 storage backend expects either a 'secret' mapping "
                "or the legacy keys: "
                + ", ".join(required_keys)
                + f". Missing: {missing}"
            )
        secret_params = {
            "KEY_ID": config["aws_access_key_id"],
            "SECRET": config["aws_secret_access_key"],
            "ENDPOINT": config["endpoint_url"],
            "URL_STYLE": config.get("url_style", "path"),
            "REGION": config.get("region", "us-east-1"),
            "USE_SSL": config.get("use_ssl", True),
            "SCOPE": config.get("scope") or f"s3://{config['bucket']}",
        }
    secret_sql = _build_secret_sql(secret_name, "S3", secret_params)

    data_path = config.get("data_path")
    if not data_path:
        bucket = config.get("bucket")
        prefix = config.get("prefix")
        if bucket:
            clean_prefix = str(prefix or "").strip("/")
            base_path = f"s3://{bucket}"
            data_path = (
                f"{base_path}/{clean_prefix}/" if clean_prefix else f"{base_path}/"
            )
        else:
            scope = secret_params.get("SCOPE")
            if isinstance(scope, str) and scope.startswith("s3://"):
                data_path = scope if scope.endswith("/") else f"{scope}/"
    if not data_path:
        raise ValueError(
            "DuckLake S3 storage backend requires either 'data_path', a 'bucket', "
            "or a secret SCOPE starting with 's3://'."
        )

    data_path_sql = f"DATA_PATH {_stringify_scalar(data_path)}"
    return secret_sql, data_path_sql


def _build_secret_sql(
    secret_name: str, secret_type: str, parameters: Mapping[str, Any]
) -> str:
    formatted_params = _format_secret_parameters(parameters)
    extra_clause = f", {', '.join(formatted_params)}" if formatted_params else ""
    return (
        f"CREATE OR REPLACE SECRET {secret_name} ( TYPE {secret_type}{extra_clause} );"
    )


def _format_secret_parameters(parameters: Mapping[str, Any]) -> list[str]:
    parts: list[str] = []
    for key, value in sorted(parameters.items()):
        formatted = _stringify_scalar(value)
        if formatted is None:
            continue
        parts.append(f"{key} {formatted}")
    return parts


def _stringify_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _format_attach_options(options: Mapping[str, Any] | None) -> str:
    if not options:
        return ""

    parts: list[str] = []
    for key, value in sorted(options.items()):
        formatted = _stringify_scalar(value)
        if formatted is None:
            continue
        parts.append(f"{str(key).upper()} {formatted}")

    return f" ({', '.join(parts)})" if parts else ""


@dataclass
class DuckLakeAttachmentConfig:
    """Configuration payload used to attach DuckLake to a DuckDB connection."""

    metadata_backend: DuckLakeBackend
    storage_backend: DuckLakeBackend
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

    def configure(self, conn: DuckDBConnection | _PreviewConnection) -> None:
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

            metadata_secret_sql, metadata_params_sql = _resolve_metadata_backend(
                self._config.metadata_backend, self._config.alias
            )
            storage_secret_sql, storage_params_sql = _resolve_storage_backend(
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
        metadata_backend: DuckLakeBackendInput,
        storage_backend: DuckLakeBackendInput,
        alias: str = "ducklake",
        plugins: Iterable[str] | None = None,
        attach_options: Mapping[str, Any] | None = None,
        extensions: Sequence[ExtensionSpec | str] | None = None,
        database: str = ":memory:",
        config: Mapping[str, str] | None = None,
        **kwargs: Any,
    ):
        metadata_cfg = _coerce_backend_config(metadata_backend, role="metadata backend")
        storage_cfg = _coerce_backend_config(storage_backend, role="storage backend")

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
        attachment = DuckLakeAttachmentConfig(
            metadata_backend=metadata_cfg,
            storage_backend=storage_cfg,
            alias=alias,
            plugins=plugin_list,
            attach_options=dict(attach_options or {}),
        )
        self._ducklake_attachment = DuckLakeAttachmentManager(attachment)

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
