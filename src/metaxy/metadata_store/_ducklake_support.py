"""Shared DuckLake configuration helpers."""

import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from duckdb import DuckDBPyConnection  # noqa: TID252
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
)

from metaxy._decorators import public


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


def coerce_backend_config(backend: DuckLakeBackendInput, *, role: str) -> DuckLakeBackend:
    """Normalize metadata/storage backend configuration."""
    if isinstance(backend, SupportsDuckLakeParts):
        return backend
    if isinstance(backend, SupportsModelDump):
        return dict(backend.model_dump())
    if isinstance(backend, Mapping):
        return dict(backend)
    raise TypeError(
        f"DuckLake {role} must be a mapping or expose get_ducklake_sql_parts()/model_dump(), got {type(backend)!r}."
    )


def resolve_metadata_backend(backend: DuckLakeBackend, alias: str) -> tuple[str, str]:
    """Generate DuckLake metadata backend SQL fragments."""
    if isinstance(backend, SupportsDuckLakeParts):
        return backend.get_ducklake_sql_parts(alias)
    return _metadata_sql_from_mapping(backend, alias)


def resolve_storage_backend(backend: DuckLakeBackend, alias: str) -> tuple[str, str]:
    """Generate DuckLake storage backend SQL fragments."""
    if isinstance(backend, SupportsDuckLakeParts):
        return backend.get_ducklake_sql_parts(alias)
    return _storage_sql_from_mapping(backend, alias)


def _metadata_sql_from_mapping(config: Mapping[str, Any], alias: str) -> tuple[str, str]:
    backend_type = str(config.get("type", "")).lower()
    if backend_type == "postgres":
        return _metadata_postgres_sql(config, alias)
    if backend_type in {"sqlite", "duckdb"}:
        path = config.get("path")
        if not path:
            raise ValueError(f"DuckLake metadata backend of type '{backend_type}' requires a 'path' entry.")
        literal_path = _stringify_scalar(path)
        return "", f"METADATA_PATH {literal_path}"
    raise ValueError(f"Unsupported DuckLake metadata backend type: {backend_type!r}")


def _metadata_postgres_sql(config: Mapping[str, Any], alias: str) -> tuple[str, str]:
    database = config.get("database")
    user = config.get("user")
    password = config.get("password")
    if database is None or user is None or password is None:
        raise ValueError("DuckLake postgres metadata backend requires 'database', 'user', and 'password'.")
    host = config.get("host") or os.getenv("DUCKLAKE_PG_HOST", "localhost")
    port_value = config.get("port", 5432)
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("DuckLake postgres metadata backend requires 'port' to be an integer.") from exc
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
    secret_sql = build_secret_sql(secret_name, "postgres", secret_params)
    metadata_params = f"METADATA_PATH '', METADATA_PARAMETERS MAP {{'TYPE': 'postgres', 'SECRET': '{secret_name}'}}"
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
                "or the legacy keys: " + ", ".join(required_keys) + f". Missing: {missing}"
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
    secret_sql = build_secret_sql(secret_name, "S3", secret_params)

    data_path = config.get("data_path")
    if not data_path:
        bucket = config.get("bucket")
        prefix = config.get("prefix")
        if bucket:
            clean_prefix = str(prefix or "").strip("/")
            base_path = f"s3://{bucket}"
            data_path = f"{base_path}/{clean_prefix}/" if clean_prefix else f"{base_path}/"
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


def build_secret_sql(secret_name: str, secret_type: str, parameters: Mapping[str, Any]) -> str:
    """Construct DuckDB secret creation SQL."""
    formatted_params = _format_secret_parameters(parameters)
    extra_clause = f", {', '.join(formatted_params)}" if formatted_params else ""
    return f"CREATE OR REPLACE SECRET {secret_name} ( TYPE {secret_type}{extra_clause} );"


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


def format_attach_options(options: Mapping[str, Any] | None) -> str:
    """Format ATTACH options clause."""
    if not options:
        return ""

    parts: list[str] = []
    for key, value in sorted(options.items()):
        formatted = _stringify_scalar(value)
        if formatted is None:
            continue
        parts.append(f"{str(key).upper()} {formatted}")

    return f" ({', '.join(parts)})" if parts else ""


@public
class DuckLakeAttachmentConfig(BaseModel):
    """Configuration payload used to attach DuckLake to a DuckDB connection."""

    metadata_backend: DuckLakeBackend
    storage_backend: DuckLakeBackend
    alias: str = "ducklake"
    plugins: tuple[str, ...] = Field(default_factory=lambda: ("ducklake",))
    attach_options: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @field_validator("metadata_backend", "storage_backend", mode="before")
    @classmethod
    def _coerce_backends(cls, value: DuckLakeBackendInput, info: ValidationInfo) -> DuckLakeBackend:
        field_name = info.field_name or "backend"
        return coerce_backend_config(value, role=field_name.replace("_", " "))

    @field_validator("alias", mode="before")
    @classmethod
    def _coerce_alias(cls, value: Any) -> str:
        if value is None:
            return "ducklake"
        alias = str(value).strip()
        return alias or "ducklake"

    @field_validator("plugins", mode="before")
    @classmethod
    def _coerce_plugins(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ("ducklake",)
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence):
            try:
                return tuple(str(item) for item in value)
            except TypeError as exc:  # pragma: no cover - defensive guard
                raise TypeError("DuckLake plugins must be a string or sequence of strings.") from exc
        raise TypeError("DuckLake plugins must be a string or sequence of strings.")

    @field_validator("attach_options", mode="before")
    @classmethod
    def _coerce_attach_options(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("DuckLake attach_options must be a mapping if provided.")

    @computed_field(return_type=tuple[str, str])
    def metadata_sql_parts(self) -> tuple[str, str]:
        """Pre-computed metadata SQL components for DuckLake attachments."""
        return resolve_metadata_backend(self.metadata_backend, self.alias)

    @computed_field(return_type=tuple[str, str])
    def storage_sql_parts(self) -> tuple[str, str]:
        """Pre-computed storage SQL components for DuckLake attachments."""
        return resolve_storage_backend(self.storage_backend, self.alias)


class _PreviewCursor:
    """Collect commands for previewing DuckLake attachment SQL."""

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

    def configure(self, conn: DuckDBPyConnection | _PreviewConnection) -> None:
        cursor = conn.cursor()
        try:
            for plugin in self._config.plugins:
                cursor.execute(f"INSTALL {plugin};")
                cursor.execute(f"LOAD {plugin};")

            metadata_secret_sql, metadata_params_sql = self._config.metadata_sql_parts
            storage_secret_sql, storage_params_sql = self._config.storage_sql_parts

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

            options_clause = format_attach_options(self._config.attach_options)
            cursor.execute(f"ATTACH 'ducklake:{ducklake_secret}' AS {self._config.alias}{options_clause};")
            cursor.execute(f"USE {self._config.alias};")
        finally:
            cursor.close()

    def preview_sql(self) -> list[str]:
        """Return the SQL statements that would be executed during configure()."""
        preview_conn = _PreviewConnection()
        self.configure(preview_conn)
        return preview_conn.cursor().commands


DuckLakeConfigInput = DuckLakeAttachmentConfig | Mapping[str, Any]


def build_ducklake_attachment(
    config: DuckLakeConfigInput,
) -> tuple[DuckLakeAttachmentConfig, DuckLakeAttachmentManager]:
    """Normalise ducklake configuration and create attachment manager."""
    if isinstance(config, DuckLakeAttachmentConfig):
        attachment_config = config
    elif isinstance(config, Mapping):
        attachment_config = DuckLakeAttachmentConfig.model_validate(config)
    else:  # pragma: no cover - defensive programming
        raise TypeError("DuckLake configuration must be a DuckLakeAttachmentConfig or mapping.")

    manager = DuckLakeAttachmentManager(attachment_config)
    return attachment_config, manager


def ensure_extensions_with_plugins(
    extensions: list[str | Any],  # list[str | ExtensionSpec] - ExtensionSpec from duckdb.py
    plugins: Sequence[str],
) -> None:
    """Ensure DuckLake plugins are present in the extensions list.

    Args:
        extensions: List of extension names (str) or ExtensionSpec objects
        plugins: DuckLake plugin names to ensure are in the extensions list
    """
    existing_names: set[str] = set()
    for ext in extensions:
        if isinstance(ext, str):
            existing_names.add(ext)
        elif isinstance(ext, Mapping):
            name = ext.get("name")
            if not name:
                raise ValueError(f"DuckDB extension mapping must have a non-empty 'name' key, got: {ext!r}")
            existing_names.add(str(name))
        else:
            # Must be ExtensionSpec with 'name' attribute
            existing_names.add(ext.name)

    for plugin in plugins:
        if plugin not in existing_names:
            extensions.append(plugin)
            existing_names.add(plugin)
