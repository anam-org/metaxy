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


class DuckDBMetadataBackendConfig(BaseModel):
    """DuckDB file-based metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["duckdb"] = "duckdb"
    uri: str

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH {_stringify_scalar(self.uri)}"


class SQLiteMetadataBackendConfig(BaseModel):
    """SQLite file-based metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["sqlite"] = "sqlite"
    uri: str

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        return "", f"METADATA_PATH {_stringify_scalar(self.uri)}"


class PostgresMetadataBackendConfig(BaseModel):
    """PostgreSQL metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["postgres"] = "postgres"
    database: str
    user: str
    password: str
    host: str
    port: int = 5432
    secret_parameters: dict[str, Any] | None = None

    def sql_parts(self, alias: str) -> tuple[str, str]:
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

        secret_name = f"secret_catalog_{alias}"
        secret_sql = build_secret_sql(secret_name, "postgres", secret_params)
        metadata_params = f"METADATA_PATH '', METADATA_PARAMETERS MAP {{'TYPE': 'postgres', 'SECRET': '{secret_name}'}}"
        return secret_sql, metadata_params


class MotherDuckMetadataBackendConfig(BaseModel):
    """[MotherDuck](https://motherduck.com/)-managed metadata backend for [DuckLake](https://ducklake.select/)."""

    type: Literal["motherduck"] = "motherduck"
    database: str


# ---------------------------------------------------------------------------
# Storage backend configs
# ---------------------------------------------------------------------------


class LocalStorageBackendConfig(BaseModel):
    """Local filesystem storage backend for DuckLake."""

    type: Literal["local"] = "local"
    path: str

    def sql_parts(self, _alias: str) -> tuple[str, str]:
        return "", f"DATA_PATH {_stringify_scalar(self.path)}"


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
    secret_parameters: dict[str, Any] | None = None

    def sql_parts(self, alias: str) -> tuple[str, str]:
        secret_name = f"secret_storage_{alias}"
        secret_params: dict[str, Any] = {}

        if self.key_id is not None and self.secret is not None:
            secret_params["KEY_ID"] = self.key_id
            secret_params["SECRET"] = self.secret
        else:
            secret_params["PROVIDER"] = "credential_chain"

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

        secret_sql = build_secret_sql(secret_name, "S3", secret_params)

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

        return secret_sql, f"DATA_PATH {_stringify_scalar(data_path)}"


class R2StorageBackendConfig(BaseModel):
    """Cloudflare R2 storage backend for [DuckLake](https://ducklake.select/).

    Uses the DuckDB ``TYPE R2`` secret which requires an ``ACCOUNT_ID``.
    """

    type: Literal["r2"] = "r2"
    key_id: str | None = None
    secret: str | None = None
    account_id: str
    data_path: str
    secret_parameters: dict[str, Any] | None = None

    def sql_parts(self, alias: str) -> tuple[str, str]:
        secret_name = f"secret_storage_{alias}"
        secret_params: dict[str, Any] = {"ACCOUNT_ID": self.account_id}

        if self.key_id is not None and self.secret is not None:
            secret_params["KEY_ID"] = self.key_id
            secret_params["SECRET"] = self.secret
        else:
            secret_params["PROVIDER"] = "credential_chain"

        if self.secret_parameters:
            for key, value in self.secret_parameters.items():
                secret_params[str(key).upper()] = value

        secret_sql = build_secret_sql(secret_name, "R2", secret_params)
        return secret_sql, f"DATA_PATH {_stringify_scalar(self.data_path)}"


class GCSStorageBackendConfig(BaseModel):
    """Google Cloud Storage backend for [DuckLake](https://ducklake.select/).

    Uses the DuckDB ``TYPE GCS`` secret with HMAC authentication.
    """

    type: Literal["gcs"] = "gcs"
    key_id: str | None = None
    secret: str | None = None
    data_path: str
    secret_parameters: dict[str, Any] | None = None

    def sql_parts(self, alias: str) -> tuple[str, str]:
        secret_name = f"secret_storage_{alias}"
        secret_params: dict[str, Any] = {}

        if self.key_id is not None and self.secret is not None:
            secret_params["KEY_ID"] = self.key_id
            secret_params["SECRET"] = self.secret
        else:
            secret_params["PROVIDER"] = "credential_chain"

        if self.secret_parameters:
            for key, value in self.secret_parameters.items():
                secret_params[str(key).upper()] = value

        secret_sql = build_secret_sql(secret_name, "GCS", secret_params)
        return secret_sql, f"DATA_PATH {_stringify_scalar(self.data_path)}"


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


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

    def __init__(self, config: DuckLakeAttachmentConfig):
        self._config = config
        self._attached: bool = False

    def _build_sql_statements(self) -> list[str]:
        """Build the full list of SQL statements for DuckLake attachment (secrets + ATTACH)."""
        statements: list[str] = []

        if isinstance(self._config.metadata_backend, MotherDuckMetadataBackendConfig):
            return self._build_motherduck_statements(statements)

        return self._build_standard_statements(statements)

    def _build_motherduck_statements(self, statements: list[str]) -> list[str]:
        """Build SQL statements for MotherDuck-managed DuckLake."""
        assert isinstance(self._config.metadata_backend, MotherDuckMetadataBackendConfig)
        db = self._config.metadata_backend.database
        alias = self._config.alias
        options_clause = format_attach_options(self._config.attach_options)
        statements.append(f"ATTACH IF NOT EXISTS 'ducklake:md:__ducklake_metadata_{db}' AS {alias}{options_clause};")
        statements.append(f"USE {alias};")
        return statements

    def _build_standard_statements(self, statements: list[str]) -> list[str]:
        """Build SQL statements for self-managed DuckLake backends."""
        metadata_backend = self._config.metadata_backend
        assert isinstance(
            metadata_backend, DuckDBMetadataBackendConfig | SQLiteMetadataBackendConfig | PostgresMetadataBackendConfig
        )
        assert self._config.storage_backend is not None
        metadata_secret_sql, metadata_params_sql = metadata_backend.sql_parts(self._config.alias)
        storage_secret_sql, storage_params_sql = self._config.storage_backend.sql_parts(self._config.alias)

        if metadata_secret_sql:
            statements.append(metadata_secret_sql)
        if storage_secret_sql:
            statements.append(storage_secret_sql)

        ducklake_secret = f"secret_{self._config.alias}"
        statements.append(
            f"CREATE OR REPLACE SECRET {ducklake_secret} ("
            " TYPE DUCKLAKE,"
            f" {metadata_params_sql},"
            f" {storage_params_sql}"
            " );"
        )

        options_clause = format_attach_options(self._config.attach_options)
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
