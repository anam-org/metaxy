"""ClickHouse metadata store - thin wrapper around IbisMetadataStore."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.ibis import IbisMetadataStore


class ClickHouseMetadataStore(IbisMetadataStore):
    """
    ClickHouse metadata store using Ibis backend.

    Convenience wrapper that configures IbisMetadataStore for ClickHouse.

    Hash algorithm support:
    - MD5: Always available (built-in)
    - XXHASH32, XXHASH64: Available via ClickHouse's xxHash32/xxHash64 functions

    Components:
        - joiner: NarwhalsJoiner (works with any backend)
        - calculator: IbisDataVersionCalculator (native SQL hash computation with xxHash64/xxHash32/MD5)
        - diff_resolver: NarwhalsDiffResolver

    Examples:
        >>> # Local ClickHouse instance
        >>> with ClickHouseMetadataStore("clickhouse://localhost:9000/default") as store:
        ...     store.write_metadata(MyFeature, df)

        >>> # With authentication
        >>> with ClickHouseMetadataStore("clickhouse://user:pass@host:9000/db") as store:
        ...     store.write_metadata(MyFeature, df)

        >>> # Using connection params
        >>> store = ClickHouseMetadataStore(
        ...     backend="clickhouse",
        ...     connection_params={
        ...         "host": "localhost",
        ...         "port": 9000,
        ...         "database": "default",
        ...         "user": "default",
        ...         "password": ""
        ...     },
        ...     hash_algorithm=HashAlgorithm.XXHASH64
        ... )
        >>> with store:
        ...     store.write_metadata(MyFeature, df)
    """

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict | None = None,
        fallback_stores: list["MetadataStore"] | None = None,
        **kwargs,
    ):
        """
        Initialize ClickHouse metadata store.

        Args:
            connection_string: ClickHouse connection string.
                Format: "clickhouse://[user[:password]@]host[:port]/database[?param=value]"
                Examples:
                    - "clickhouse://localhost:9000/default"
                    - "clickhouse://user:pass@host:9000/db"
                    - "clickhouse://host:9000/db?secure=true"
            connection_params: Alternative to connection_string, specify params as dict:
                - host: Server host (default: "localhost")
                - port: Server port (default: 9000)
                - database: Database name (default: "default")
                - user: Username (default: "default")
                - password: Password (default: "")
                - secure: Use secure connection (default: False)
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Passed to IbisMetadataStore (e.g., hash_algorithm, graph)

        Raises:
            ImportError: If ibis-clickhouse not installed
            ValueError: If neither connection_string nor connection_params provided
        """
        if connection_string is None and connection_params is None:
            raise ValueError(
                "Must provide either connection_string or connection_params. "
                "Example: connection_string='clickhouse://localhost:9000/default'"
            )

        # Initialize Ibis store with ClickHouse backend
        super().__init__(
            connection_string=connection_string,
            backend="clickhouse" if connection_string is None else None,
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for ClickHouse stores.

        Uses XXHASH64 which is built-in to ClickHouse.
        """
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """ClickHouse stores support native components when connection is open."""
        return self._conn is not None

    def _get_hash_sql_generators(self) -> dict[HashAlgorithm, "HashSQLGenerator"]:
        """Get hash SQL generators for ClickHouse.

        ClickHouse supports:
        - MD5: Always available (built-in)
        - XXHASH32, XXHASH64: Always available (built-in xxHash32/xxHash64 functions)

        Returns:
            Dictionary mapping HashAlgorithm to SQL generator functions
        """

        def md5_generator(table, concat_columns: dict[str, str]) -> str:
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                # Cast to String for consistency
                hash_expr = f"CAST(MD5({concat_col}) AS String)"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        def xxhash32_generator(table, concat_columns: dict[str, str]) -> str:
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                hash_expr = f"CAST(xxHash32({concat_col}) AS String)"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        def xxhash64_generator(table, concat_columns: dict[str, str]) -> str:
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                hash_expr = f"CAST(xxHash64({concat_col}) AS String)"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        return {
            HashAlgorithm.MD5: md5_generator,
            HashAlgorithm.XXHASH32: xxhash32_generator,
            HashAlgorithm.XXHASH64: xxhash64_generator,
        }
