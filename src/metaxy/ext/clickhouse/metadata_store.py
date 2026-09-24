"""This module implements [`IbisMetadataStore`][metaxy.ext.ibis.metadata_store.IbisMetadataStore] for ClickHouse.

It takes care of some ClickHouse-specific logic such as native hash functions and `Map(K, V)` type handling."""

from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
from narwhals.typing import FrameT

if TYPE_CHECKING:
    import ibis

    from metaxy.metadata_store.base import MetadataStore

from metaxy._decorators import public
from metaxy.ext.ibis.metadata_store import (
    IbisMetadataStore,
    IbisMetadataStoreConfig,
)
from metaxy.ext.ibis.versioning import IbisVersioningEngine
from metaxy.models.types import FeatureKey
from metaxy.versioning.types import HashAlgorithm


class ClickHouseVersioningEngine(IbisVersioningEngine):
    """Versioning engine for ClickHouse backend.

    Overrides concat_strings_over_groups to use ClickHouse-compatible
    syntax with collect() (groupArray) + arrayStringConcat.
    """

    def concat_strings_over_groups(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> FrameT:
        """Concatenate string values within groups using ClickHouse window functions.

        Uses collect() (groupArray) + arrayStringConcat instead of group_concat().over()
        which generates invalid SQL for ClickHouse.
        """
        import ibis
        import ibis.expr.datatypes as dt
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())  # ty: ignore[invalid-argument-type]

        @ibis.udf.scalar.builtin
        def arrayStringConcat(arr: dt.Array[dt.String], sep: str) -> str:  # ty: ignore[empty-body]
            """ClickHouse arrayStringConcat() function."""
            ...

        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        window = ibis.window(
            group_by=group_by_columns,
            order_by=[ibis_table[col] for col in effective_order_by],
        )

        arr_expr = ibis_table[source_column].cast("string").collect().over(window)
        concat_expr = arrayStringConcat(arr_expr, separator)

        ibis_table = ibis_table.mutate(**{target_column: concat_expr})

        return cast(FrameT, nw.from_native(ibis_table))


@public
class ClickHouseMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for ClickHouseMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.clickhouse.ClickHouseMetadataStore"

        [stores.dev.config]
        connection_string = "clickhouse://localhost:8443/default"
        hash_algorithm = "xxhash64"
        ```
    """


@public
class ClickHouseMetadataStore(IbisMetadataStore):
    """
    [ClickHouse](https://clickhouse.com/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Example: Connection Parameters
        <!-- skip next -->
        ```py
        store = ClickHouseMetadataStore(
            backend="clickhouse",
            connection_params={
                "host": "localhost",
                "port": 8443,
                "database": "default",
                "user": "default",
                "password": "",
            },
            hash_algorithm=HashAlgorithm.XXHASH64,
        )
        ```
    """

    versioning_engine_cls = ClickHouseVersioningEngine

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

            **kwargs: Passed to [`IbisMetadataStore`][metaxy.ext.ibis.metadata_store.IbisMetadataStore]`

        Raises:
            ImportError: If ibis-clickhouse not installed
            ValueError: If neither connection_string nor connection_params provided
        """
        if connection_string is None and connection_params is None:
            raise ValueError(
                "Must provide either connection_string or connection_params. "
                "Example: connection_string='clickhouse://localhost:8443/default'"
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

        Uses XXHASH32 which is built-in to ClickHouse.
        """
        return HashAlgorithm.XXHASH32

    def _create_hash_functions(self):
        """Create ClickHouse-specific hash functions for Ibis expressions.

        Implements MD5 and xxHash functions using ClickHouse's native functions.
        """
        # Import ibis for wrapping built-in SQL functions
        import ibis

        hash_functions = {}

        # ClickHouse MD5 implementation
        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:  # ty: ignore[empty-body]
            """ClickHouse MD5() function."""
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:  # ty: ignore[empty-body]
            """ClickHouse HEX() function."""
            ...

        @ibis.udf.scalar.builtin
        def lower(x: str) -> str:  # ty: ignore[empty-body]
            """ClickHouse lower() function."""
            ...

        def md5_hash(col_expr):
            """Hash a column using ClickHouse's MD5() function."""
            # MD5 returns binary FixedString(16), convert to lowercase hex
            return lower(HEX(MD5(col_expr.cast(str))))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        # ClickHouse xxHash functions
        @ibis.udf.scalar.builtin
        def xxh3(x: str) -> int:  # ty: ignore[empty-body]
            """ClickHouse xxh3() function - returns UInt64."""
            ...  # pragma: no cover

        @ibis.udf.scalar.builtin
        def xxHash32(x: str) -> int:  # ty: ignore[empty-body]
            """ClickHouse xxHash32() function - returns UInt32."""
            ...

        @ibis.udf.scalar.builtin
        def xxHash64(x: str) -> int:  # ty: ignore[empty-body]
            """ClickHouse xxHash64() function - returns UInt64."""
            ...

        @ibis.udf.scalar.builtin
        def toString(x: int) -> str:  # ty: ignore[empty-body]
            """ClickHouse toString() function - converts integer to string."""
            ...

        def xxh3_64_hash(col_expr):
            """Hash a column using ClickHouse's xxh3() function."""
            return toString(xxh3(col_expr))

        def xxhash32_hash(col_expr):
            """Hash a column using ClickHouse's xxHash32() function."""
            # xxHash32 returns UInt32, convert to string
            return toString(xxHash32(col_expr))

        def xxhash64_hash(col_expr):
            """Hash a column using ClickHouse's xxHash64() function."""
            # xxHash64 returns UInt64, convert to string
            return toString(xxHash64(col_expr))

        hash_functions[HashAlgorithm.XXH3_64] = xxh3_64_hash
        hash_functions[HashAlgorithm.XXHASH32] = xxhash32_hash
        hash_functions[HashAlgorithm.XXHASH64] = xxhash64_hash

        return hash_functions

    def transform_after_read(self, table: "ibis.Table", feature_key: "FeatureKey") -> "ibis.Table":
        """Cast ClickHouse `JSON` columns to String for PyArrow compatibility.

        The ClickHouse driver returns `JSON` columns as dicts while PyArrow expects bytes.
        `Map` columns are left as-is and collected to `polars_map.Map` by `collect_to_polars`.
        """
        import ibis.expr.datatypes as dt

        schema = table.schema()
        mutations = {
            col_name: table[col_name].cast("string") for col_name, dtype in schema.items() if isinstance(dtype, dt.JSON)
        }

        if not mutations:
            return table

        return table.mutate(**mutations)

    @staticmethod
    def _is_table_not_found_error(e: Exception) -> bool:
        import ibis.common.exceptions

        if isinstance(e, ibis.common.exceptions.TableNotFound):
            return True
        try:
            from clickhouse_connect.driver.exceptions import DatabaseError
        except ImportError:
            return False
        return isinstance(e, DatabaseError) and "UNKNOWN_TABLE" in str(e)

    def ibis_type_to_polars(self, ibis_type: Any) -> Any:
        """Convert Ibis data type to Polars data type, with ClickHouse Map support.

        Handles ``Map(K, V)`` → ``pl.List(pl.Struct({"key": K_pl, "value": V_pl}))``
        which is Arrow's canonical map representation. Delegates all other types
        to the base implementation.
        """
        import ibis.expr.datatypes as dt
        import polars as pl

        if isinstance(ibis_type, dt.Map):
            key_pl = self.ibis_type_to_polars(ibis_type.key_type)
            value_pl = self.ibis_type_to_polars(ibis_type.value_type)
            return pl.List(pl.Struct({"key": key_pl, "value": value_pl}))

        return super().ibis_type_to_polars(ibis_type)

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
        is_secure = url.port == 8443 or (url.query and url.query.get("secure") == "True")

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
