"""Shared DuckLake configuration helpers."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from duckdb import DuckDBPyConnection  # noqa: TID252
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from metaxy._decorators import public

# ---------------------------------------------------------------------------
# Metadata backend configs
# ---------------------------------------------------------------------------


@public
class DuckDBMetadataBackendConfig(BaseModel):
    """DuckDB file-based metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["duckdb"] = "duckdb"
    uri: str

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH {_stringify_scalar(self.uri)}"


@public
class SQLiteMetadataBackendConfig(BaseModel):
    """SQLite file-based metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["sqlite"] = "sqlite"
    uri: str

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH {_stringify_scalar(self.uri)}"


@public
class PostgresMetadataBackendConfig(BaseModel):
    """PostgreSQL metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["postgres"] = "postgres"
    database: str | None = None
    user: str | None = None
    password: str | None = None
    host: str | None = None
    port: int = 5432
    secret_name: str
    secret_parameters: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_credentials(self) -> Self:
        inline_fields = {"host": self.host, "database": self.database, "user": self.user, "password": self.password}
        has_inline = any(v is not None for v in inline_fields.values())
        if not has_inline:
            return self
        missing = [k for k, v in inline_fields.items() if v is None]
        if missing:
            raise ValueError(f"Missing required inline credentials: {', '.join(missing)}.")
        return self

    def _has_inline_credentials(self) -> bool:
        return any(v is not None for v in (self.host, self.database, self.user, self.password, self.secret_parameters))

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        metadata_params = (
            f"METADATA_PATH '', METADATA_PARAMETERS MAP {{'TYPE': 'postgres', 'SECRET': '{self.secret_name}'}}"
        )
        if not self._has_inline_credentials():
            return "", metadata_params

        secret_params: dict[str, Any] = {
            "HOST": self.host,
            "PORT": self.port,
            "DATABASE": self.database,
            "USER": self.user,
            "PASSWORD": self.password,
        }
        if self.secret_parameters:
            for key, value in self.secret_parameters.items():
                secret_params[str(key).upper()] = value

        secret_sql = build_secret_sql(self.secret_name, "postgres", secret_params)
        return secret_sql, metadata_params


@public
class MotherDuckMetadataBackendConfig(BaseModel):
    """[MotherDuck](https://motherduck.com/)-managed metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["motherduck"] = "motherduck"
    database: str
    region: str | None = Field(
        default=None,
        description="AWS region of the MotherDuck-managed S3 storage (e.g. 'eu-central-1').",
    )


# ---------------------------------------------------------------------------
# Storage backend configs
# ---------------------------------------------------------------------------


@public
class LocalStorageBackendConfig(BaseModel):
    """Local filesystem storage backend for DuckLake."""

    type: Literal["local"] = "local"
    path: str

    def sql_parts(self, _alias: str, *, secret_storage: str | None = None) -> tuple[str, str]:
        return "", f"DATA_PATH {_stringify_scalar(self.path)}"


@public
class S3StorageBackendConfig(BaseModel):
    """S3 storage backend for DuckLake."""

    type: Literal["s3"] = "s3"
    key_id: str | None = None
    secret: str | None = None
    endpoint: str | None = None
    bucket: str | None = None
    prefix: str | None = None
    region: str | None = None
    url_style: str | None = None
    use_ssl: bool | None = None
    scope: str | None = None
    data_path: str | None = None
    secret_name: str
    secret_parameters: dict[str, Any] | None = None

    def _resolve_data_path(self) -> str:
        data_path = self.data_path
        if not data_path:
            if self.bucket:
                clean_prefix = str(self.prefix or "").strip("/")
                base_path = f"s3://{self.bucket}"
                data_path = f"{base_path}/{clean_prefix}/" if clean_prefix else f"{base_path}/"
            elif self.scope and self.scope.startswith("s3://"):
                data_path = self.scope if self.scope.endswith("/") else f"{self.scope}/"
        if not data_path:
            raise ValueError(
                "DuckLake S3 storage backend requires either 'data_path', a 'bucket', or 'scope' starting with 's3://'."
            )
        return data_path

    def _has_secret_config_fields(self) -> bool:
        return any(
            v is not None
            for v in (
                self.key_id,
                self.secret,
                self.endpoint,
                self.region,
                self.url_style,
                self.use_ssl,
                self.scope,
                self.secret_parameters,
            )
        )

    def sql_parts(self, _alias: str, *, secret_storage: str | None = None) -> tuple[str, str]:
        data_path_sql = f"DATA_PATH {_stringify_scalar(self._resolve_data_path())}"

        if not self._has_secret_config_fields():
            return "", data_path_sql

        secret_params: dict[str, Any] = {}

        if self.key_id is not None and self.secret is not None:
            secret_params["KEY_ID"] = self.key_id
            secret_params["SECRET"] = self.secret

        if self.endpoint is not None:
            endpoint = self.endpoint
            # DuckDB ENDPOINT expects host:port without scheme; USE_SSL controls http vs https
            if endpoint.startswith("https://"):
                endpoint = endpoint.removeprefix("https://")
                if self.use_ssl is None:
                    secret_params["USE_SSL"] = True
            elif endpoint.startswith("http://"):
                endpoint = endpoint.removeprefix("http://")
                if self.use_ssl is None:
                    secret_params["USE_SSL"] = False
            secret_params["ENDPOINT"] = endpoint
        if self.region is not None:
            secret_params["REGION"] = self.region
        if self.url_style is not None:
            secret_params["URL_STYLE"] = self.url_style
        if self.use_ssl is not None:
            secret_params["USE_SSL"] = self.use_ssl
        if self.scope is not None:
            secret_params["SCOPE"] = self.scope
        if self.secret_parameters:
            for key, value in self.secret_parameters.items():
                secret_params[str(key).upper()] = value

        secret_sql = build_secret_sql(self.secret_name, "S3", secret_params, secret_storage=secret_storage)
        return secret_sql, data_path_sql


@public
class R2StorageBackendConfig(BaseModel):
    """Cloudflare R2 storage backend for [DuckLake](https://ducklake.select/).

    Uses the DuckDB ``TYPE R2`` secret which requires an ``ACCOUNT_ID``.
    """

    type: Literal["r2"] = "r2"
    key_id: str | None = None
    secret: str | None = None
    account_id: str | None = None
    data_path: str
    secret_name: str
    secret_parameters: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_credentials(self) -> Self:
        has_inline = self.key_id is not None or self.secret is not None
        if has_inline and self.account_id is None:
            raise ValueError("'account_id' is required when providing inline credentials (key_id, secret).")
        return self

    def _has_secret_config_fields(self) -> bool:
        return any(v is not None for v in (self.key_id, self.secret, self.account_id, self.secret_parameters))

    def sql_parts(self, _alias: str, *, secret_storage: str | None = None) -> tuple[str, str]:
        data_path_sql = f"DATA_PATH {_stringify_scalar(self.data_path)}"

        if not self._has_secret_config_fields():
            return "", data_path_sql

        secret_params: dict[str, Any] = {}
        if self.account_id is not None:
            secret_params["ACCOUNT_ID"] = self.account_id

        if self.key_id is not None and self.secret is not None:
            secret_params["KEY_ID"] = self.key_id
            secret_params["SECRET"] = self.secret

        if self.secret_parameters:
            for key, value in self.secret_parameters.items():
                secret_params[str(key).upper()] = value

        secret_sql = build_secret_sql(self.secret_name, "R2", secret_params, secret_storage=secret_storage)
        return secret_sql, data_path_sql


@public
class GCSStorageBackendConfig(BaseModel):
    """Google Cloud Storage backend for [DuckLake](https://ducklake.select/).

    Uses the DuckDB ``TYPE GCS`` secret with HMAC authentication.
    """

    type: Literal["gcs"] = "gcs"
    key_id: str | None = None
    secret: str | None = None
    data_path: str
    secret_name: str
    secret_parameters: dict[str, Any] | None = None

    def _has_secret_config_fields(self) -> bool:
        return any(v is not None for v in (self.key_id, self.secret, self.secret_parameters))

    def sql_parts(self, _alias: str, *, secret_storage: str | None = None) -> tuple[str, str]:
        data_path_sql = f"DATA_PATH {_stringify_scalar(self.data_path)}"

        if not self._has_secret_config_fields():
            return "", data_path_sql

        secret_params: dict[str, Any] = {}

        if self.key_id is not None and self.secret is not None:
            secret_params["KEY_ID"] = self.key_id
            secret_params["SECRET"] = self.secret

        if self.secret_parameters:
            for key, value in self.secret_parameters.items():
                secret_params[str(key).upper()] = value

        secret_sql = build_secret_sql(self.secret_name, "GCS", secret_params, secret_storage=secret_storage)
        return secret_sql, data_path_sql


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def build_secret_sql(
    secret_name: str, secret_type: str, parameters: Mapping[str, Any], *, secret_storage: str | None = None
) -> str:
    """Construct DuckDB secret creation SQL."""
    storage_clause = f" IN {secret_storage}" if secret_storage else ""
    formatted_params = _format_secret_parameters(parameters)
    extra_clause = f", {', '.join(formatted_params)}" if formatted_params else ""
    return f"CREATE OR REPLACE SECRET {secret_name}{storage_clause} ( TYPE {secret_type}{extra_clause} );"


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


# ---------------------------------------------------------------------------
# Attachment config
# ---------------------------------------------------------------------------


@public
class DuckLakeAttachmentConfig(BaseModel):
    """[DuckLake](https://ducklake.select/) attachment configuration for a DuckDB connection."""

    metadata_backend: Annotated[
        DuckDBMetadataBackendConfig
        | SQLiteMetadataBackendConfig
        | PostgresMetadataBackendConfig
        | MotherDuckMetadataBackendConfig,
        Field(discriminator="type"),
    ] = Field(description="Metadata catalog backend (DuckDB, SQLite, PostgreSQL, or MotherDuck).")
    storage_backend: (
        Annotated[
            LocalStorageBackendConfig | S3StorageBackendConfig | R2StorageBackendConfig | GCSStorageBackendConfig,
            Field(discriminator="type"),
        ]
        | None
    ) = Field(
        default=None,
        description="Data storage backend (local filesystem, S3, R2, or GCS). Not required for MotherDuck.",
    )
    alias: str = Field(default="ducklake", description="DuckDB catalog alias for the attached DuckLake database.")
    attach_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra [DuckLake](https://ducklake.select/) ATTACH options (e.g., api_version, override_data_path).",
    )
    data_inlining_row_limit: int | None = Field(
        default=None,
        description="Store inserts smaller than this row count directly in the metadata catalog "
        "instead of creating Parquet files.",
    )

    @model_validator(mode="after")
    def _validate_storage_backend(self) -> Self:
        if not isinstance(self.metadata_backend, MotherDuckMetadataBackendConfig) and self.storage_backend is None:
            raise ValueError("storage_backend is required for non-MotherDuck metadata backends.")
        return self

    @field_validator("alias", mode="before")
    @classmethod
    def _coerce_alias(cls, value: Any) -> str:
        if value is None:
            return "ducklake"
        alias = str(value).strip()
        return alias or "ducklake"

    @field_validator("attach_options", mode="before")
    @classmethod
    def _coerce_attach_options(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("DuckLake attach_options must be a mapping if provided.")


# ---------------------------------------------------------------------------
# Attachment manager
# ---------------------------------------------------------------------------


class DuckLakeAttachmentManager:
    """Responsible for configuring a DuckDB connection for DuckLake usage."""

    def __init__(self, config: DuckLakeAttachmentConfig, *, store_name: str | None = None):
        self._config = config
        self._secret_suffix = f"{store_name}_{config.alias}" if store_name else config.alias
        self._attached: bool = False

    def _build_sql_statements(self) -> list[str]:
        """Build the full list of SQL statements for DuckLake attachment (secrets + ATTACH)."""
        statements: list[str] = []

        if isinstance(self._config.metadata_backend, MotherDuckMetadataBackendConfig):
            return self._build_motherduck_statements(statements)

        return self._build_standard_statements(statements)

    def _build_motherduck_statements(self, statements: list[str]) -> list[str]:
        """Build SQL statements for MotherDuck-managed DuckLake.

        Fully managed MotherDuck DuckLake databases are available to any
        MotherDuck connection via ``USE``.  When a ``storage_backend`` is
        provided (BYOB mode), the DuckLake database is created with a custom
        ``DATA_PATH`` and storage secrets are created ``IN MOTHERDUCK``.
        """
        assert isinstance(self._config.metadata_backend, MotherDuckMetadataBackendConfig)
        md = self._config.metadata_backend
        if md.region is not None:
            statements.append(f"SET s3_region='{md.region}';")
        if self._config.storage_backend is not None:
            storage_secret_sql, data_path_sql = self._config.storage_backend.sql_parts(
                self._secret_suffix, secret_storage="MOTHERDUCK"
            )
            statements.append(f"CREATE DATABASE IF NOT EXISTS {md.database} (TYPE DUCKLAKE, {data_path_sql});")
            if storage_secret_sql:
                statements.append(storage_secret_sql)
        statements.append(f"USE {md.database};")
        return statements

    def _build_standard_statements(self, statements: list[str]) -> list[str]:
        """Build SQL statements for self-managed DuckLake backends."""
        metadata_backend = self._config.metadata_backend
        assert isinstance(
            metadata_backend, DuckDBMetadataBackendConfig | SQLiteMetadataBackendConfig | PostgresMetadataBackendConfig
        )
        assert self._config.storage_backend is not None
        metadata_secret_sql, metadata_params_sql = metadata_backend.sql_parts(self._secret_suffix)
        storage_secret_sql, storage_params_sql = self._config.storage_backend.sql_parts(self._secret_suffix)

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
