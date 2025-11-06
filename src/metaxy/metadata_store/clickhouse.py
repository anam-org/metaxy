"""ClickHouse metadata store - thin wrapper around IbisMetadataStore."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store.ibis import IbisMetadataStore


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
        """ClickHouse stores support native field provenance calculations when connection is open."""
        return self._conn is not None

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
