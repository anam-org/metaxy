"""This module implements [`IbisMetadataStore`][metaxy.metadata_store.ibis.IbisMetadataStore] for ClickHouse.

It takes care of some ClickHouse-specific logic such as `nw.Struct` type conversion against ClickHouse types such as `Map(K,V)`."""

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
            connection_string="clickhouse://localhost:8443/default",
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

        store = ClickHouseMetadataStore.from_config(config)
        ```
    """

    pass  # All fields inherited from IbisMetadataStoreConfig


class ClickHouseMetadataStore(IbisMetadataStore):
    """
    [ClickHouse](https://clickhouse.com/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Example: Connection Parameters
        ```py
        store = ClickHouseMetadataStore(
            backend="clickhouse",
            connection_params={
                "host": "localhost",
                "port": 8443,
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

                Example:
                    ```
                    "clickhouse://localhost:8443/default"
                    ```

            connection_params: Alternative to connection_string, specify params as dict:

                - host: Server host

                - port: Server port (default: `8443`)

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
                "Example: connection_string='clickhouse://localhost:8443/default'"
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
        def MD5(x: str) -> str:  # ty: ignore[invalid-return-type]
            """ClickHouse MD5() function."""
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:  # ty: ignore[invalid-return-type]
            """ClickHouse HEX() function."""
            ...

        @ibis.udf.scalar.builtin
        def lower(x: str) -> str:  # ty: ignore[invalid-return-type]
            """ClickHouse lower() function."""
            ...

        def md5_hash(col_expr):
            """Hash a column using ClickHouse's MD5() function."""
            # MD5 returns binary FixedString(16), convert to lowercase hex
            return lower(HEX(MD5(col_expr.cast(str))))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        # ClickHouse xxHash functions
        @ibis.udf.scalar.builtin
        def xxHash32(x: str) -> int:  # ty: ignore[invalid-return-type]
            """ClickHouse xxHash32() function - returns UInt32."""
            ...

        @ibis.udf.scalar.builtin
        def xxHash64(x: str) -> int:  # ty: ignore[invalid-return-type]
            """ClickHouse xxHash64() function - returns UInt64."""
            ...

        @ibis.udf.scalar.builtin
        def toString(x: int) -> str:  # ty: ignore[invalid-return-type]
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

        - `JSON` columns: Cast to String (ClickHouse driver returns dict, PyArrow expects bytes)

        - `Map(String, String)` metaxy columns: Convert to named Struct by extracting keys

        For metaxy Map columns (`metaxy_provenance_by_field`, `metaxy_data_version_by_field`),
        we build a named Struct from map key accesses using known field names from the
        feature spec.

        User-defined Map columns are left as-is and will appear in e.g. Polars as
        `List[Struct{key, value}]` (the standard Arrow Map representation).
        """
        import ibis.expr.datatypes as dt

        from metaxy.models.constants import (
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE_BY_FIELD,
        )

        # Only convert these metaxy system Map columns to Struct
        metaxy_map_columns = {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}

        schema = table.schema()
        mutations: dict[str, Any] = {}

        for col_name, dtype in schema.items():
            if isinstance(dtype, dt.JSON):
                # JSON → String (can't convert to Struct due to ClickHouse CAST limitations)
                mutations[col_name] = table[col_name].cast("string")

            elif isinstance(dtype, dt.Map) and col_name in metaxy_map_columns:
                # Only convert metaxy system Map(String, String) columns to Struct
                # User-defined Map columns are left as-is
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
        struct_dict = {name: map_col.get(name, "") for name in field_names}

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
        """Transform Struct columns to Map-compatible format for ClickHouse.

        Only transforms known metaxy struct columns (metaxy_provenance_by_field,
        metaxy_data_version_by_field) when the ClickHouse table has Map columns.

        For Polars: Converts Struct to List[Struct{key, value}] which Ibis/ClickHouse
        recognizes as array<struct<key, value>> and can insert into Map(K,V) columns.

        For Ibis: Converts Struct to Map using ibis.map() function.

        Args:
            df: Input DataFrame (may be Narwhals wrapping Polars or Ibis)
            ch_schema: ClickHouse table schema

        Returns:
            DataFrame with Struct columns converted to Map-compatible format
        """
        import ibis.expr.datatypes as dt

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

        # Handle Ibis-backed DataFrames (keep lazy)
        if df.implementation == nw.Implementation.IBIS:
            return self._transform_ibis_struct_to_map(df, map_columns)

        # All other backends: collect to Polars and transform
        return self._transform_polars_struct_to_map(df, map_columns)

    def _transform_ibis_struct_to_map(self, df: Frame, map_columns: set[str]) -> Frame:
        """Transform Ibis Struct columns to Map format for ClickHouse.

        Args:
            df: Narwhals DataFrame backed by Ibis
            map_columns: Set of column names that need to be converted to Map

        Returns:
            DataFrame with Struct columns converted to Map
        """
        from typing import cast as typing_cast

        import ibis
        import ibis.expr.datatypes as dt

        ibis_table = typing_cast("ibis.Table", df.to_native())
        schema = ibis_table.schema()

        mutations: dict[str, ibis.Expr] = {}
        for col_name in map_columns:
            if col_name not in schema:
                continue

            col_dtype = schema[col_name]
            if not isinstance(col_dtype, dt.Struct):
                continue

            # Get field names from the struct type
            field_names = list(col_dtype.names)

            # Build map from struct fields: Map(field_name -> field_value)
            # ibis.map() takes two arrays: keys and values
            keys = ibis.array([ibis.literal(name) for name in field_names])
            values = ibis.array([ibis_table[col_name][name] for name in field_names])
            mutations[col_name] = ibis.map(keys, values)

        if not mutations:
            return df

        result_table = ibis_table.mutate(**mutations)  # ty: ignore[invalid-argument-type]
        return nw.from_native(result_table, eager_only=False)

    def _transform_polars_struct_to_map(
        self, df: Frame, map_columns: set[str]
    ) -> Frame:
        """Transform Polars Struct columns to Map-compatible format for ClickHouse.

        Args:
            df: Narwhals DataFrame backed by Polars
            map_columns: Set of column names that need to be converted to Map

        Returns:
            DataFrame with Struct columns converted to List[Struct{key, value}]
        """
        import polars as pl

        from metaxy._utils import collect_to_polars

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

        return nw.from_native(pl_df)

    @property
    def sqlalchemy_url(self) -> str:
        """Get SQLAlchemy-compatible connection URL for ClickHouse.

        Overrides the base implementation to return the native protocol format
        (`clickhouse+native://`) which is required for better SQLAlchemy/Alembic
        reflection support in `clickhouse-sqlalchemy`.

        The HTTP protocol used by Ibis has [limited reflection
        capabilities](https://github.com/xzkostyan/clickhouse-sqlalchemy/issues/15).

        Port mapping (assumes default ports):

        - HTTP `8123` (non-secure) → Native `9000`

        - HTTP `8443` (secure) → Native `9440`

        For secure connections, adds `secure=True` query parameter.

        Returns:
            SQLAlchemy-compatible URL string with native protocol

        Raises:
            ValueError: If connection_string is not available
        """
        from sqlalchemy.engine.url import make_url

        base_url = super().sqlalchemy_url
        url = make_url(base_url)

        # Determine if secure based on port or existing secure param
        is_secure = url.port == 8443 or (
            url.query and url.query.get("secure") == "True"
        )

        # Map HTTP ports to native ports
        if url.port == 8443:
            native_port = 9440
        elif url.port == 8123:
            native_port = 9000
        else:
            # Non-standard port - assume secure if original was secure
            native_port = 9440 if is_secure else 9000

        # Build new URL with native protocol
        url = url.set(
            drivername="clickhouse+native",
            port=native_port,
        )

        # Handle query parameters - add secure=True for secure connections
        if is_secure:
            # Remove protocol=https (HTTP-specific) and ensure secure=True
            new_query = {k: v for k, v in (url.query or {}).items() if k != "protocol"}
            new_query["secure"] = "True"
            url = url.set(query=new_query)

        return url.render_as_string(hide_password=False)

    @classmethod
    def config_model(cls) -> type[ClickHouseMetadataStoreConfig]:
        return ClickHouseMetadataStoreConfig
