"""ClickHouse metadata store with automatic type adaptation.

Automatically converts Polars Struct columns to ClickHouse Map format
based on the actual database schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
from narwhals.typing import Frame

if TYPE_CHECKING:
    import polars as pl

    from metaxy.metadata_store.base import MetadataStore
    from metaxy.models.types import FeatureKey

from metaxy.metadata_store.ibis import IbisMetadataStore, IbisMetadataStoreConfig
from metaxy.versioning.types import HashAlgorithm


class ClickHouseMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for ClickHouseMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from IbisMetadataStoreConfig.

    Example:
        ```python
        config = ClickHouseMetadataStoreConfig(
            connection_string="clickhouse://localhost:9000/default",
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

        store = ClickHouseMetadataStore.from_config(config)
        ```
    """

    pass  # All fields inherited from IbisMetadataStoreConfig


class ClickHouseMetadataStore(IbisMetadataStore):
    """
    [ClickHouse](https://clickhouse.com/) metadata storeusing [Ibis](https://ibis-project.org/) backend.

    Example: Connection Parameters
        ```py
        store = ClickHouseMetadataStore(
            backend="clickhouse",
            connection_params={
                "host": "localhost",
                "port": 9000,
                "database": "default",
                "user": "default",
                "password": ""
            },
            hash_algorithm=HashAlgorithm.XXHASH64
        )
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize [ClickHouse](https://clickhouse.com/) metadata store.

        Args:
            connection_string: ClickHouse connection string.

                Format: `clickhouse://[user[:password]@]host[:port]/database[?param=value]`

                Examples:
                    ```
                    - "clickhouse://localhost:9000/default"
                    - "clickhouse://user:pass@host:9000/db"
                    - "clickhouse://host:9000/db?secure=true"
                    ```

            connection_params: Alternative to connection_string, specify params as dict:

                - host: Server host

                - port: Server port (default: `9000`)

                - database: Database name

                - user: Username

                - password: Password

                - secure: Use secure connection (default: `False`)

            fallback_stores: Ordered list of read-only fallback stores.

            **kwargs: Passed to [metaxy.metadata_store.ibis.IbisMetadataStore][]`

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

        # Cache for Map column lookups per table
        self._map_columns_cache: dict[str, set[str]] = {}

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for ClickHouse stores.

        Uses XXHASH64 which is built-in to ClickHouse.
        """
        return HashAlgorithm.XXHASH64

    def _create_hash_functions(self):
        """Create ClickHouse-specific hash functions for Ibis expressions.

        Implements MD5 and xxHash functions using ClickHouse's native functions.
        """
        # Import ibis for wrapping built-in SQL functions
        import ibis

        hash_functions = {}

        # ClickHouse MD5 implementation
        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:
            """ClickHouse MD5() function."""
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:
            """ClickHouse HEX() function."""
            ...

        @ibis.udf.scalar.builtin
        def lower(x: str) -> str:
            """ClickHouse lower() function."""
            ...

        def md5_hash(col_expr):
            """Hash a column using ClickHouse's MD5() function."""
            # MD5 returns binary FixedString(16), convert to lowercase hex
            return lower(HEX(MD5(col_expr.cast(str))))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        # ClickHouse xxHash functions
        @ibis.udf.scalar.builtin
        def xxHash32(x: str) -> int:
            """ClickHouse xxHash32() function - returns UInt32."""
            ...

        @ibis.udf.scalar.builtin
        def xxHash64(x: str) -> int:
            """ClickHouse xxHash64() function - returns UInt64."""
            ...

        @ibis.udf.scalar.builtin
        def toString(x: int) -> str:
            """ClickHouse toString() function - converts integer to string."""
            ...

        def xxhash32_hash(col_expr):
            """Hash a column using ClickHouse's xxHash32() function."""
            # xxHash32 returns UInt32, convert to string
            return toString(xxHash32(col_expr))

        def xxhash64_hash(col_expr):
            """Hash a column using ClickHouse's xxHash64() function."""
            # xxHash64 returns UInt64, convert to string
            return toString(xxHash64(col_expr))

        hash_functions[HashAlgorithm.XXHASH32] = xxhash32_hash
        hash_functions[HashAlgorithm.XXHASH64] = xxhash64_hash

        return hash_functions

    @classmethod
    def config_model(cls) -> type[ClickHouseMetadataStoreConfig]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return ClickHouseMetadataStoreConfig

    def _get_map_columns(self, table_name: str) -> set[str]:
        """Get columns that are Map type in the ClickHouse schema.

        Results are cached per table for performance.

        Args:
            table_name: Name of the table to inspect

        Returns:
            Set of column names that have Map type in ClickHouse.
            Returns empty set if table doesn't exist yet.
        """
        import ibis.expr.datatypes as dt
        from ibis.common.exceptions import TableNotFound

        if table_name not in self._map_columns_cache:
            try:
                schema = self.conn.get_schema(table_name)  # pyright: ignore[reportAttributeAccessIssue]
                map_cols = {
                    name for name, dtype in schema.items() if isinstance(dtype, dt.Map)
                }
            except TableNotFound:
                # Table doesn't exist yet (will be auto-created), no Map columns to convert
                map_cols = set()
            self._map_columns_cache[table_name] = map_cols

        return self._map_columns_cache[table_name]

    def _convert_struct_to_map(
        self, df: pl.DataFrame, map_columns: set[str]
    ) -> pl.DataFrame:
        """Convert Polars Struct columns to Map-compatible format.

        ClickHouse Map expects data as tuple of arrays: ([keys], [values]).
        This converts Polars Struct {k1: v1, k2: v2} to that format.

        Args:
            df: Polars DataFrame to transform
            map_columns: Set of column names that need Struct -> Map conversion

        Returns:
            DataFrame with Struct columns converted to Map-compatible format
        """
        import polars as pl

        columns_to_convert = []
        for col in map_columns:
            if col in df.columns:
                dtype = df.schema.get(col)
                if isinstance(dtype, pl.Struct):
                    columns_to_convert.append(col)

        if not columns_to_convert:
            return df

        # Convert each Struct column to Map format
        conversions = []
        for col in columns_to_convert:
            struct_dtype = cast(pl.Struct, df.schema[col])
            field_names = [f.name for f in struct_dtype.fields]
            # Create a map from struct: extract keys and values as parallel lists
            conversions.append(
                pl.struct(
                    keys=pl.lit(field_names),
                    values=pl.concat_list(
                        [pl.col(col).struct.field(f) for f in field_names]
                    ),
                ).alias(col)
            )

        # Replace the struct columns with map-compatible format
        other_cols = [pl.col(c) for c in df.columns if c not in columns_to_convert]
        return df.select(other_cols + conversions)

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write metadata to ClickHouse with automatic type conversion.

        Converts Polars Struct columns to ClickHouse Map format based on
        the actual database schema.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated)
            **kwargs: Backend-specific parameters
        """
        if df.implementation == nw.Implementation.POLARS:
            table_name = self.get_table_name(feature_key)
            map_columns = self._get_map_columns(table_name)

            if map_columns:  # Ibis does not properly convert pl.Struct into the format expected by ClickHouse's MAP(K,V)
                native_df = df.lazy().collect().to_polars()
                converted_df = self._convert_struct_to_map(native_df, map_columns)
                df = nw.from_native(converted_df)

        super().write_metadata_to_store(feature_key, df, **kwargs)
