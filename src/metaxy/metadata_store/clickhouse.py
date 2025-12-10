"""ClickHouse metadata store - thin wrapper around IbisMetadataStore."""

from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    import ibis
    from ibis.expr.schema import Schema as IbisSchema

    from metaxy.metadata_store.base import MetadataStore

from metaxy.metadata_store.ibis import (
    Frame,
    IbisMetadataStore,
    IbisMetadataStoreConfig,
)
from metaxy.models.types import FeatureKey
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
        fallback_stores: list["MetadataStore"] | None = None,
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

        # Cache for ClickHouse table schemas (cleared on close)
        self._ch_schema_cache: dict[str, IbisSchema] = {}

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

    def _get_cached_schema(self, table_name: str) -> "IbisSchema":
        """Get cached ClickHouse table schema, fetching if not cached.

        Args:
            table_name: Name of the table

        Returns:
            Ibis schema for the table
        """
        if table_name not in self._ch_schema_cache:
            self._ch_schema_cache[table_name] = self.conn.table(table_name).schema()
        return self._ch_schema_cache[table_name]

    def transform_after_read(
        self, table: "ibis.Table", feature_key: "FeatureKey"
    ) -> "ibis.Table":
        """Transform ClickHouse-specific column types for PyArrow compatibility.

        Handles:
        - JSON columns: Cast to String (ClickHouse driver returns dict, PyArrow expects bytes)
        - Map(K,V) columns: Convert to Struct by extracting keys

        For Map columns, we use ClickHouse's `tuple()` function to build a named tuple
        from map key accesses, which Ibis/PyArrow converts to Struct.
        """
        import ibis.expr.datatypes as dt

        schema = table.schema()
        mutations: dict[str, Any] = {}

        for col_name, dtype in schema.items():
            if isinstance(dtype, dt.JSON):
                # JSON → String (can't convert to Struct due to ClickHouse CAST limitations)
                mutations[col_name] = table[col_name].cast("string")

            elif isinstance(dtype, dt.Map):
                # Map(K,V) → Struct by extracting all keys from the map
                # We need to know what keys exist - get them from the actual data
                # using mapKeys() and building a struct dynamically
                mutations[col_name] = self._map_to_struct_expr(
                    table, col_name, dtype, feature_key
                )

        if not mutations:
            return table

        return table.mutate(**mutations)

    def _map_to_struct_expr(
        self,
        table: "ibis.Table",
        col_name: str,
        map_dtype: Any,  # dt.Map - avoid generic type param issues
        feature_key: "FeatureKey",
    ) -> Any:
        """Convert a Map column to Struct expression.

        ClickHouse Map(String, String) can be converted to a Struct by
        extracting specific keys using ibis.struct().

        Args:
            table: Ibis table
            col_name: Map column name
            map_dtype: Map data type (has key_type, value_type)
            feature_key: Feature key to get field names from

        Returns:
            Ibis expression that produces a Struct
        """
        import ibis

        from metaxy.models.feature import FeatureGraph

        # Get field names from the feature spec
        graph = FeatureGraph.get_active()
        if feature_key not in graph.features_by_key:
            # Feature not in graph - fall back to String cast
            return table[col_name].cast("string")

        feature_cls = graph.features_by_key[feature_key]
        # Use to_struct_key() for struct field names (uses "_" separator, not "/")
        # This matches how provenance/data_version fields are accessed elsewhere
        field_names = [f.key.to_struct_key() for f in feature_cls.spec().fields]

        if not field_names:
            return table[col_name].cast("string")

        # Build Struct from map key access using safe access via .get()
        # This returns empty string for missing keys instead of throwing KeyError
        # This is essential for empty tables/maps where keys don't exist yet
        map_col = table[col_name]
        struct_dict = {name: map_col.get(name, "") for name in field_names}  # pyright: ignore[reportAttributeAccessIssue]

        return ibis.struct(struct_dict)

    def transform_before_write(
        self, df: Frame, feature_key: "FeatureKey", table_name: str
    ) -> Frame:
        """Transform Polars Struct columns to Map format for ClickHouse.

        If the ClickHouse table has Map(K,V) columns but the DataFrame has Struct
        columns, convert the Struct to Map format before inserting.
        """
        # Check if table exists and get its schema
        if table_name not in self.conn.list_tables():
            return df

        ch_schema = self._get_cached_schema(table_name)
        return self._transform_struct_to_map(df, ch_schema)

    def _transform_struct_to_map(self, df: Frame, ch_schema: "IbisSchema") -> Frame:
        """Transform Polars Struct columns to Map-compatible format for ClickHouse.

        Only transforms known metaxy struct columns (metaxy_provenance_by_field,
        metaxy_data_version_by_field) when the ClickHouse table has Map columns.

        Converts Polars Struct to List[Struct{key, value}] which Ibis/ClickHouse
        recognizes as array<struct<key, value>> and can insert into Map(K,V) columns.

        Args:
            df: Input DataFrame (may be Narwhals wrapping Polars)
            ch_schema: ClickHouse table schema

        Returns:
            DataFrame with Struct columns converted to Map-compatible format
        """
        import ibis.expr.datatypes as dt
        import polars as pl

        from metaxy._utils import collect_to_polars
        from metaxy.models.constants import (
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE_BY_FIELD,
        )

        # Only handle known metaxy struct columns
        metaxy_struct_columns = {
            METAXY_PROVENANCE_BY_FIELD,
            METAXY_DATA_VERSION_BY_FIELD,
        }

        # Find Map columns in ClickHouse schema that are metaxy struct columns
        map_columns = {
            name
            for name, dtype in ch_schema.items()
            if isinstance(dtype, dt.Map) and name in metaxy_struct_columns
        }

        if not map_columns:
            return df

        # Only transform if DataFrame is Polars-backed
        if df.implementation != nw.Implementation.POLARS:
            return df

        # Get native Polars DataFrame
        pl_df = collect_to_polars(df)

        # Check which columns need transformation (are Struct in Polars)
        cols_to_transform: list[tuple[str, list[str]]] = []
        for col_name in map_columns:
            if col_name in pl_df.columns:
                col_dtype = pl_df.schema[col_name]
                if isinstance(col_dtype, pl.Struct):
                    field_names = [f.name for f in col_dtype.fields]
                    cols_to_transform.append((col_name, field_names))

        if not cols_to_transform:
            return df

        # Transform Struct columns to List[Struct{key, value}] format
        # This is what Ibis/ClickHouse expects for Map(K,V) columns
        transformations = []
        for col_name, field_names in cols_to_transform:
            # Build list of {key: field_name, value: field_value} structs
            key_value_structs = [
                pl.struct(
                    [
                        pl.lit(field_name).alias("key"),
                        pl.col(col_name).struct.field(field_name).alias("value"),
                    ]
                )
                for field_name in field_names
            ]
            transformations.append(pl.concat_list(key_value_structs).alias(col_name))

        pl_df = pl_df.with_columns(transformations)

        return nw.from_native(pl_df, eager_only=False)

    @classmethod
    def config_model(cls) -> type[ClickHouseMetadataStoreConfig]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return ClickHouseMetadataStoreConfig
