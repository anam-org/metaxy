"""DuckLake configuration models and SQL helpers."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

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
class DuckDBCatalogConfig(BaseModel):
    """DuckDB file-based metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["duckdb"] = "duckdb"
    uri: str

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH {_stringify_scalar(self.uri)}"


@public
class SQLiteCatalogConfig(BaseModel):
    """SQLite file-based metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["sqlite"] = "sqlite"
    uri: str

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH {_stringify_scalar(self.uri)}"


@public
class PostgresCatalogConfig(BaseModel):
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
class MotherDuckCatalogConfig(BaseModel):
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
class LocalStorageConfig(BaseModel):
    """Local filesystem storage backend for DuckLake."""

    type: Literal["local"] = "local"
    path: str

    def sql_parts(self, _alias: str, *, secret_storage: str | None = None) -> tuple[str, str]:
        return "", f"DATA_PATH {_stringify_scalar(self.path)}"


@public
class S3StorageConfig(BaseModel):
    """[S3 storage](https://duckdb.org/docs/stable/core_extensions/httpfs/s3api) backend for DuckLake."""

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
class R2StorageConfig(BaseModel):
    """Cloudflare R2 storage backend for [DuckLake](https://ducklake.select/).

    Uses the DuckDB [`TYPE R2`](https://duckdb.org/docs/stable/core_extensions/httpfs/s3api#r2-secrets) secret.
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
class GCSStorageConfig(BaseModel):
    """Google Cloud Storage backend for [DuckLake](https://ducklake.select/).

    Uses the DuckDB [`TYPE GCS`](https://duckdb.org/docs/stable/core_extensions/httpfs/s3api#gcs-secrets) secret.
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
class DuckLakeConfig(BaseModel):
    """[DuckLake](https://ducklake.select/) attachment configuration for a DuckDB connection."""

    catalog: Annotated[
        DuckDBCatalogConfig | SQLiteCatalogConfig | PostgresCatalogConfig | MotherDuckCatalogConfig,
        Field(discriminator="type"),
    ] = Field(description="Metadata catalog backend (DuckDB, SQLite, PostgreSQL, or MotherDuck).")
    storage: (
        Annotated[
            LocalStorageConfig | S3StorageConfig | R2StorageConfig | GCSStorageConfig,
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
    def _validate_storage(self) -> Self:
        if not isinstance(self.catalog, MotherDuckCatalogConfig) and self.storage is None:
            raise ValueError("storage is required for non-MotherDuck metadata backends.")
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
