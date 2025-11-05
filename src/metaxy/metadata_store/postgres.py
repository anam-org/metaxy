"""PostgreSQL metadata store - thin wrapper around IbisMetadataStore."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.ibis import IbisMetadataStore


class PostgresMetadataStore(IbisMetadataStore):
    """
    [PostgreSQL](https://www.postgresql.org/) metadata store using [Ibis](https://ibis-project.org/) backend.

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
            schema="public",
        )
        ```

    Example: Custom Hash Algorithm
        ```py
        # Requires pgcrypto extension for SHA256 support
        store = PostgresMetadataStore(
            "postgresql://user:pass@localhost:5432/metadata",
            hash_algorithm=HashAlgorithm.SHA256,
        )
        ```
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
        fallback_stores: list["MetadataStore"] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize [PostgreSQL](https://www.postgresql.org/) metadata store.

        Args:
            connection_string: PostgreSQL connection string.
                Format: ``postgresql://user:pass@host:port/database``.
            host: Server host (used when connection_string not provided).
            port: Server port (defaults to 5432 when omitted).
            user: Database user.
            password: Database password.
            database: Database name.
            schema: Target schema (defaults to user's search_path when omitted).
            connection_params: Additional Ibis PostgreSQL connection parameters.
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Passed to [metaxy.metadata_store.ibis.IbisMetadataStore][].

        Raises:
            ValueError: If neither connection_string nor connection parameters provided.
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

        super().__init__(
            connection_string=connection_string,
            backend="postgres" if connection_string is None else None,
            connection_params=params if connection_string is None else params or None,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    def _get_hash_sql_generators(self) -> dict[HashAlgorithm, "HashSQLGenerator"]:
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
