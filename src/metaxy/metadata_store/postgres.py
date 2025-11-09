"""PostgreSQL metadata store - thin wrapper around IbisMetadataStore.

Provides production-grade metadata storage using PostgreSQL with:
- Full ACID compliance
- Schema isolation
- Extension support (pgcrypto for SHA256)
- Connection pooling support
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
import polars as pl

if TYPE_CHECKING:
    from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.feature import BaseFeature
    from metaxy.models.types import FeatureKey

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.base import PROVENANCE_BY_FIELD_COL
from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError
from metaxy.metadata_store.ibis import IbisMetadataStore

logger = logging.getLogger(__name__)


class PostgresMetadataStore(IbisMetadataStore):
    """
    [PostgreSQL](https://www.postgresql.org/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Provides production-grade metadata storage with full ACID compliance, schema isolation,
    and support for advanced hash algorithms via PostgreSQL extensions.

    **Hash Algorithm Support:**

    - **MD5** (default): Built-in, no extensions required
    - **SHA256**: Requires `pgcrypto` extension (auto-enabled on open)

    The store automatically enables `pgcrypto` when using SHA256 hashing.

    **Schema Isolation:**

    Use the `schema` parameter to isolate metadata in a dedicated schema.
    If not specified, tables are created in the user's search_path (typically `public`).

    Example: Connection String
        ```py
        store = PostgresMetadataStore("postgresql://user:pass@localhost:5432/metadata")
        ```

    Example: Connection Parameters
        ```py
        store = PostgresMetadataStore(
            host="localhost",
            port=5432,
            user="ml",
            password="secret",
            database="metaxy",
            schema="features",  # Optional schema isolation
        )
        ```

    Example: SHA256 Hash Algorithm
        ```py
        # pgcrypto extension auto-enabled
        store = PostgresMetadataStore(
            "postgresql://user:pass@localhost:5432/metadata",
            hash_algorithm=HashAlgorithm.SHA256,
        )
        ```

    Example: With Fallback Stores
        ```py
        # Read from prod, write to dev
        prod_store = PostgresMetadataStore(
            "postgresql://user:pass@prod:5432/metadata",
        )
        dev_store = PostgresMetadataStore(
            "postgresql://user:pass@dev:5432/metadata",
            fallback_stores=[prod_store],
        )
        ```

    Warning:
        SHA256 requires the `pgcrypto` extension. The store will attempt to enable it
        automatically on open. Ensure your database user has CREATE EXTENSION privileges,
        or have a DBA pre-create the extension.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        schema: str | None = None,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        enable_pgcrypto: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize [PostgreSQL](https://www.postgresql.org/) metadata store.

        Args:
            connection_string: PostgreSQL connection string.
                Format: `postgresql://user:pass@host:port/database`.
                Supports additional parameters: `?options=-c%20statement_timeout=30s`.
            host: Server host (used when connection_string not provided).
            port: Server port (defaults to 5432 when omitted).
            user: Database user.
            password: Database password.
            database: Database name.
            schema: Target schema for table isolation (defaults to search_path when omitted).
                Recommended for production deployments.
            connection_params: Additional Ibis PostgreSQL connection parameters
                (e.g., `{"sslmode": "require", "connect_timeout": 10}`).
            fallback_stores: Ordered list of read-only fallback stores for branch deployments.
            enable_pgcrypto: Whether to auto-enable pgcrypto extension on open (default: True).
                Set to False if pgcrypto is already enabled or if user lacks CREATE EXTENSION.
            **kwargs: Passed to [metaxy.metadata_store.ibis.IbisMetadataStore][]
                (e.g., `hash_algorithm`, `auto_create_tables`).

        Raises:
            ValueError: If neither connection_string nor connection parameters provided.
            ImportError: If Ibis or psycopg2 driver not installed.

        Note:
            When using SHA256 hash algorithm, pgcrypto extension is required.
            The store attempts to enable it automatically on open unless
            `enable_pgcrypto=False`.
        """
        params: dict[str, Any] = dict(connection_params or {})

        explicit_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "schema": schema,
        }
        for key, value in explicit_params.items():
            if value is not None:
                params.setdefault(key, value)

        if connection_string is None and not params:
            raise ValueError(
                "Must provide either connection_string or connection parameters. "
                "Example: connection_string='postgresql://user:pass@localhost:5432/db' "
                "or host='localhost', database='db'."
            )

        if connection_string is None and "port" not in params:
            params["port"] = 5432

        self.host = params.get("host")
        self.port = params.get("port")
        self.database = params.get("database")
        self.schema = params.get("schema")
        self.enable_pgcrypto = enable_pgcrypto

        super().__init__(
            connection_string=connection_string,
            backend="postgres" if connection_string is None else None,
            connection_params=params if connection_string is None else params or None,
            fallback_stores=fallback_stores,
            **kwargs,
        )

        supported_algorithms = {HashAlgorithm.MD5, HashAlgorithm.SHA256}
        if self.hash_algorithm not in supported_algorithms:
            raise HashAlgorithmNotSupportedError(
                f"PostgresMetadataStore supports only MD5 and SHA256 hash algorithms. "
                f"Requested: {self.hash_algorithm}"
            )

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for PostgreSQL stores.

        Uses MD5 as it's built-in and requires no extensions.
        For SHA256 support, explicitly set hash_algorithm=HashAlgorithm.SHA256
        and ensure pgcrypto extension is enabled.
        """
        return HashAlgorithm.MD5

    def open(self) -> None:
        """Open connection to PostgreSQL and enable required extensions.

        If using SHA256 hash algorithm and enable_pgcrypto=True, attempts to
        enable the pgcrypto extension. Logs a warning if extension cannot be enabled.

        Raises:
            ImportError: If psycopg2 driver not installed.
            Various database errors: If connection fails.
        """
        super().open()

        if not self._has_native_struct_support():
            # Postgres + current Ibis/sqlglot versions cannot create STRUCT columns.
            # Fall back to JSON serialization + in-memory Narwhals/Polars processing.
            self._struct_compat_mode = True
            logger.warning(
                "PostgreSQL backend lacks native STRUCT type support in this environment. "
                "Falling back to JSON serialization and in-memory projections. "
                "For better performance, upgrade to a version of Ibis/sqlglot that "
                "supports STRUCT types on Postgres."
            )
        else:
            self._struct_compat_mode = False

        # Auto-enable pgcrypto if SHA256 is configured
        if self.enable_pgcrypto and self.hash_algorithm == HashAlgorithm.SHA256:
            self._ensure_pgcrypto_extension()

    def _ensure_pgcrypto_extension(self) -> None:
        """Ensure pgcrypto extension is enabled for SHA256 support.

        Attempts to create the extension if it doesn't exist. Logs a warning
        if the user lacks privileges rather than failing, as the extension
        might already be enabled.
        """
        try:
            # Use underlying connection to execute DDL (Ibis doesn't expose CREATE EXTENSION)
            # For PostgreSQL backend, conn.con is the psycopg2 connection
            raw_conn = self.conn.con  # pyright: ignore[reportAttributeAccessIssue]
            with raw_conn.cursor() as cursor:  # pyright: ignore[reportAttributeAccessIssue]
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                raw_conn.commit()  # pyright: ignore[reportAttributeAccessIssue]
            logger.debug("pgcrypto extension enabled successfully")
        except Exception as e:
            # Log warning but don't fail - extension might already be enabled
            # or user might lack privileges but extension is globally enabled
            logger.warning(
                f"Could not enable pgcrypto extension: {e}. "
                "If using SHA256 hash algorithm, ensure pgcrypto is enabled "
                "or have a DBA run: CREATE EXTENSION IF NOT EXISTS pgcrypto;"
            )

    def _get_hash_sql_generators(self) -> dict[HashAlgorithm, HashSQLGenerator]:
        """Get hash SQL generators for PostgreSQL."""
        generators = super()._get_hash_sql_generators()

        def sha256_generator(table, concat_columns: dict[str, str]) -> str:
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                hash_expr = f"ENCODE(DIGEST({concat_col}, 'sha256'), 'hex')"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        generators[HashAlgorithm.SHA256] = sha256_generator
        return generators

    def _supports_native_components(self) -> bool:
        """Disable native components when running in struct-compatibility mode."""
        return super()._supports_native_components() and not self._struct_compat_mode

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """Serialize struct columns when Postgres lacks STRUCT support."""
        if self._struct_compat_mode and PROVENANCE_BY_FIELD_COL in df.columns:
            df = self._serialize_provenance_column(df)
        super()._write_metadata_impl(feature_key, df)

    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata, deserializing provenance structs when needed."""
        lazy_frame = super().read_metadata_in_store(
            feature,
            feature_version=feature_version,
            filters=filters,
            columns=columns,
        )

        if not self._struct_compat_mode or lazy_frame is None:
            return lazy_frame

        native = lazy_frame.to_native()
        execute = getattr(native, "execute", None)
        if execute is None:
            # Unexpected backend (already materialized) - return as-is
            return lazy_frame

        native_result = execute()
        if isinstance(native_result, pl.DataFrame):
            polars_df: pl.DataFrame = native_result
        else:
            try:
                polars_df = cast(
                    pl.DataFrame,
                    pl.from_arrow(native_result),  # pyarrow.Table or RecordBatchReader
                )
            except Exception:
                polars_df = pl.from_pandas(native_result)

        polars_df = self._deserialize_provenance_column(polars_df)
        return nw.from_native(polars_df.lazy(), eager_only=False)

    def read_metadata_in_store(self, feature, **kwargs):
        """Ensure dependency data_version_by_field columns exist after unpack."""

        from metaxy.models.constants import (
            METAXY_DATA_VERSION,
            METAXY_PROVENANCE,
        )

        lf = super().read_metadata_in_store(feature, **kwargs)
        if lf is None:
            return lf

        feature_key = self._resolve_feature_key(feature)
        plan = self._resolve_feature_plan(feature_key)

        # Ensure flattened provenance/data_version columns exist for this feature's fields
        expected_fields = [field.key.to_struct_key() for field in plan.feature.fields]
        for field_name in expected_fields:
            prov_flat = f"{METAXY_PROVENANCE_BY_FIELD}__{field_name}"
            if prov_flat not in lf.columns:
                if METAXY_PROVENANCE in lf.columns:
                    lf = lf.with_columns(nw.col(METAXY_PROVENANCE).alias(prov_flat))
                else:
                    lf = lf.with_columns(nw.lit(None, dtype=nw.String).alias(prov_flat))

            data_flat = f"{METAXY_DATA_VERSION_BY_FIELD}__{field_name}"
            if data_flat not in lf.columns:
                if METAXY_DATA_VERSION in lf.columns:
                    lf = lf.with_columns(nw.col(METAXY_DATA_VERSION).alias(data_flat))
                else:
                    lf = lf.with_columns(nw.lit(None, dtype=nw.String).alias(data_flat))

        return lf

    def write_metadata_to_store(self, feature_key: FeatureKey, df, **kwargs):
        """Ensure string-typed materialization_id before delegating write."""

        import polars as pl

        from metaxy.models.constants import METAXY_MATERIALIZATION_ID

        if METAXY_MATERIALIZATION_ID in df.columns:
            df = df.with_columns(nw.col(METAXY_MATERIALIZATION_ID).cast(nw.String))

        # Postgres rejects NULL-typed columns; cast Null columns to string for Polars inputs
        if df.implementation == nw.Implementation.POLARS:
            null_cols = [
                col
                for col, dtype in df.schema.items()
                if dtype == pl.Null  # type: ignore[attr-defined]
            ]
            if null_cols:
                df = df.with_columns(
                    [nw.col(col).cast(nw.String).alias(col) for col in null_cols]
                )

        super().write_metadata_to_store(feature_key, df, **kwargs)

    def read_metadata(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version=None,
        filters=None,
        columns=None,
        allow_fallback=True,
        current_only=True,
        latest_only=True,
    ):
        """Read metadata using the flattened JSON-compatible layout."""
        return super().read_metadata(
            feature,
            feature_version=feature_version,
            filters=filters,
            columns=columns,
            allow_fallback=allow_fallback,
            current_only=current_only,
            latest_only=latest_only,
        )

    def display(self) -> str:
        """Display string for this store."""
        details: list[str] = []
        if self.database:
            details.append(f"database={self.database}")
        if self.schema:
            details.append(f"schema={self.schema}")
        if self.host:
            details.append(f"host={self.host}")
        if self.port:
            details.append(f"port={self.port}")

        if self._is_open:
            details.append(f"features={len(self._list_features_local())}")

        detail_str = ", ".join(details)
        if detail_str:
            return f"PostgresMetadataStore({detail_str})"
        if self.connection_string:
            return f"PostgresMetadataStore(connection_string={self.connection_string})"
        return "PostgresMetadataStore()"

    def _has_native_struct_support(self) -> bool:
        """Detect whether current Ibis/sqlglot stack can compile STRUCT types."""
        try:
            from sqlglot import exp
        except Exception:  # pragma: no cover - only triggered if sqlglot missing
            return False

        dialect = getattr(self.conn, "dialect", None)
        type_mapping = getattr(dialect, "TYPE_TO_EXPRESSIONS", None)
        if not type_mapping:
            return False

        data_type_cls = getattr(exp, "DataType", None)
        type_enum = getattr(data_type_cls, "Type", None) if data_type_cls else None
        struct_type = getattr(type_enum, "STRUCT", None) if type_enum else None
        if struct_type is None:
            return False

        return struct_type in type_mapping

    @staticmethod
    def _serialize_provenance_column(df: pl.DataFrame) -> pl.DataFrame:
        """Convert struct provenance column to canonical JSON for storage."""
        column = df.get_column(PROVENANCE_BY_FIELD_COL)
        serialized = [
            json.dumps(value, sort_keys=True) if value is not None else None
            for value in column.to_list()
        ]
        return df.with_columns(
            pl.Series(name=PROVENANCE_BY_FIELD_COL, values=serialized, dtype=pl.String)
        )

    @staticmethod
    def _deserialize_provenance_column(df: pl.DataFrame) -> pl.DataFrame:
        """Restore struct provenance column from JSON text."""
        if PROVENANCE_BY_FIELD_COL not in df.columns:
            return df
        column = df.get_column(PROVENANCE_BY_FIELD_COL)
        decoded = [
            json.loads(value) if value is not None else None
            for value in column.to_list()
        ]
        return df.with_columns(pl.Series(name=PROVENANCE_BY_FIELD_COL, values=decoded))
