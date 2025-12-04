"""ClickHouse metadata store - thin wrapper around IbisMetadataStore."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import ibis

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy.metadata_store.ibis import IbisMetadataStore, IbisMetadataStoreConfig
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
            mutations_sync=2,  # Wait for all replicas
        )

        store = ClickHouseMetadataStore.from_config(config)
        ```
    """

    mutations_sync: int = 1
    """Consistency level for DELETE/UPDATE mutations.

    ClickHouse mutations (ALTER TABLE UPDATE/DELETE) are asynchronous by default.
    This setting controls synchronization behavior:

    - `0`: Asynchronous (fire-and-forget). Fastest, but changes may not be immediately visible.
    - `1`: Wait for mutation to complete on this replica (default). Good balance of speed and consistency.
    - `2`: Wait for mutation to complete on all replicas. Strongest consistency, slowest.

    See: https://clickhouse.com/docs/en/operations/settings/settings#mutations_sync
    """


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
        mutations_sync: int = 1,
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

            mutations_sync: Consistency level for DELETE/UPDATE mutations (0=async, 1=this replica, 2=all replicas).

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

        # Store mutations_sync setting
        self._mutations_sync = mutations_sync

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

    @contextmanager
    def open(self, mode="read"):  # noqa: ANN001
        """Open ClickHouse connection and set mutation settings at session level."""
        with super().open(mode) as store:
            try:
                cast(Any, self.conn).raw_sql(
                    f"SET mutations_sync={self._mutations_sync}"
                )
            except Exception:
                # Best-effort; avoid failing open on setting errors
                pass
            yield store

    def _delete_metadata_impl(
        self,
        feature_key: Any,
        filter_expr: Any,
    ) -> int:
        """Defer to generic Ibis delete implementation."""
        target_key = (
            feature_key
            if isinstance(feature_key, FeatureKey)
            else self._resolve_feature_key(feature_key)
        )
        return super()._delete_metadata_impl(target_key, filter_expr)

    def _mutate_metadata_impl(
        self,
        feature_key: Any,
        filter_expr: Any,
        updates: dict[str, Any],
    ) -> int:
        """Disable in-place UPDATE; ClickHouse mutations are append-only."""
        raise NotImplementedError(
            "Mutations are append-only for ClickHouse; in-place UPDATE disabled."
        )

    def _create_hash_functions(self):
        """Create ClickHouse-specific hash functions for Ibis expressions.

        Implements MD5 and xxHash functions using ClickHouse's native functions.
        """
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
