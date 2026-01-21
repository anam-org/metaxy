"""ADBC-based metadata store for SQL databases.

Provides high-performance bulk ingestion using Arrow Database Connectivity (ADBC).
ADBC enables zero-copy data transfer via Apache Arrow for 2-10x faster writes
compared to traditional database APIs.

Supported backends:
- PostgreSQL (adbc-driver-postgresql)
- DuckDB (adbc-driver-duckdb)
- SQLite (adbc-driver-sqlite)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame
from pydantic import Field
from typing_extensions import Self

from metaxy.metadata_store.base import (
    MetadataStore,
    MetadataStoreConfig,
    VersioningEngineOptions,
)
from metaxy.metadata_store.exceptions import StoreNotOpenError
from metaxy.metadata_store.types import AccessMode
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning import VersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    try:
        import adbc_driver_manager  # ty: ignore[unresolved-import]  # noqa: F401
    except ImportError:
        pass


class ADBCMetadataStoreConfig(MetadataStoreConfig):
    """Base configuration for ADBC metadata stores.

    Example:
        ```python
        from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

        store = ADBCPostgresMetadataStore(
            connection_string="postgresql://user:pass@host:5432/db",
            max_connections=8,
        )
        ```
    """

    connection_string: str | None = Field(
        default=None,
        description="ADBC connection URI (e.g., 'postgresql://host:5432/db').",
    )

    connection_params: dict[str, Any] | None = Field(
        default=None,
        description="Backend-specific connection parameters.",
    )

    table_prefix: str | None = Field(
        default=None,
        description="Optional prefix for all table names.",
    )

    max_connections: int = Field(
        default=1,
        ge=1,
        le=128,
        description="Maximum number of connections in the pool for concurrent writes.",
    )


class ADBCMetadataStore(MetadataStore, ABC):
    """
    Abstract base class for ADBC-backed metadata stores.

    ADBC (Arrow Database Connectivity) provides:
    - Zero-copy data transfer via Apache Arrow
    - Bulk ingestion performance (2-10x faster than traditional APIs)
    - Streaming result sets
    - Connection pooling for concurrent writes

    Storage layout:
    - Each feature gets its own table: {prefix}{feature}__{key}
    - System tables: metaxy__system__feature_versions, metaxy__system__migrations
    - Uses ADBC driver manager for cross-database compatibility

    Warning:
        This is an abstract base class. Use concrete implementations:
        - ADBCPostgresMetadataStore
        - ADBCDuckDBMetadataStore
        - ADBCSQLiteMetadataStore

    Example:
        ```py
        from metaxy.metadata_store.adbc_postgres import ADBCPostgresMetadataStore

        # PostgreSQL with connection pooling
        store = ADBCPostgresMetadataStore(
            connection_string="postgresql://user:pass@host:5432/db",
            max_connections=8,  # Pool size for bulk writes
        )

        with store:
            # Bulk write (uses connection pool)
            store.write_metadata(MyFeature, large_df)
        ```
    """

    # Subclasses must override this with their ADBC versioning engine
    versioning_engine_cls: type[VersioningEngine]

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        table_prefix: str | None = None,
        max_connections: int = 1,
        versioning_engine: VersioningEngineOptions = "auto",
        **kwargs: Any,
    ):
        """
        Initialize ADBC metadata store.

        Args:
            connection_string: ADBC connection URI (e.g., "postgresql://host:5432/db").
                If provided, connection_params are ignored.
            connection_params: Backend-specific connection parameters.
                Used when connection_string is not provided.
            table_prefix: Optional prefix applied to all table names.
            max_connections: Maximum number of connections in the pool (1-128).
                Used for concurrent bulk writes.
            versioning_engine: Which versioning engine to use.
                - "auto": Prefer the store's native engine, fall back to Polars if needed
                - "native": Always use the store's native engine
                - "polars": Always use the Polars engine
            **kwargs: Passed to MetadataStore.__init__ (e.g., fallback_stores, hash_algorithm)
        """
        super().__init__(versioning_engine=versioning_engine, **kwargs)

        self.connection_string = connection_string
        self.connection_params = connection_params or {}
        self.table_prefix = table_prefix or ""
        self.max_connections = max_connections

        # Connection state
        self._conn: Any = None  # adbc_driver_manager.AdbcConnection when open
        self._database: Any = None  # adbc_driver_manager.AdbcDatabase when open

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[MetadataStoreConfig]:
        """Return the configuration model class for this store type.

        Subclasses must override this to return their specific config class.
        """
        ...

    @abstractmethod
    def _get_driver_name(self) -> str:
        """Get the ADBC driver name for this backend.

        Returns:
            Driver name (e.g., "adbc_driver_postgresql", "adbc_driver_duckdb")
        """
        ...

    @abstractmethod
    def _get_connection_options(self) -> dict[str, Any]:
        """Get backend-specific connection options.

        Returns:
            Dictionary of connection options for the ADBC driver
        """
        ...

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get the default hash algorithm for ADBC stores.

        Returns:
            MD5 for cross-database compatibility by default.
            Subclasses can override for backend-specific algorithms.
        """
        return HashAlgorithm.MD5

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:
        """Open/initialize the ADBC connection.

        Context manager that opens the store with specified access mode.

        Args:
            mode: Access mode - "read" or "write"

        Yields:
            Self: The opened store instance

        Raises:
            StoreNotOpenError: If the store fails to connect
        """
        self._context_depth += 1

        try:
            # Only create connection on first entry
            if self._context_depth == 1:
                import adbc_driver_manager  # ty: ignore[unresolved-import]

                # Initialize database
                driver_name = self._get_driver_name()
                connection_options = self._get_connection_options()

                self._database = adbc_driver_manager.AdbcDatabase(
                    driver=driver_name,
                    **connection_options,
                )

                # Create connection from database
                self._conn = adbc_driver_manager.AdbcConnection(self._database)
                self._is_open = True

            yield self

        finally:
            self._context_depth -= 1

            # Clean up on final exit
            if self._context_depth == 0:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None

                if self._database is not None:
                    self._database = None

                self._is_open = False

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[VersioningEngine]:
        """Create versioning engine for this store.

        Args:
            plan: Feature plan containing dependency information

        Yields:
            Configured versioning engine instance

        Raises:
            StoreNotOpenError: If store is not open

        Note:
            Subclasses must override this to provide their specific engine initialization.
            For ADBC stores, this typically creates an ADBCVersioningEngine with
            backend-specific hash functions.
        """
        if not self._is_open:
            raise StoreNotOpenError("Store must be open to create versioning engine")

        # Create engine instance (subclasses will override versioning_engine_cls)
        # The exact signature depends on the engine type (e.g., ADBCVersioningEngine)
        engine = self.versioning_engine_cls(plan=plan)

        yield engine

    def _table_name(self, feature_key: FeatureKey) -> str:
        """Generate table name for a feature.

        Args:
            feature_key: Feature key

        Returns:
            Table name with optional prefix
        """
        base_name = feature_key.table_name
        return f"{self.table_prefix}{base_name}"

    def display(self) -> str:
        """Return a human-readable display string for this store.

        Returns:
            Display string showing backend and connection info
        """
        driver = self._get_driver_name()
        if self.connection_string:
            # Sanitize connection string (hide password)
            conn_str = self.connection_string
            if "@" in conn_str:
                parts = conn_str.split("@")
                if ":" in parts[0]:
                    proto_user_pass = parts[0].rsplit(":", 1)
                    proto_user = proto_user_pass[0]
                    conn_str = f"{proto_user}:****@{parts[1]}"
            return f"ADBC({driver}, {conn_str})"
        return f"ADBC({driver})"

    # Abstract methods that concrete implementations must provide

    @abstractmethod
    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Write metadata to the store using ADBC bulk ingestion.

        Args:
            feature_key: Feature identifier
            df: Narwhals DataFrame to write
            **kwargs: Backend-specific options
        """
        ...

    @abstractmethod
    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Any = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from the store.

        Args:
            feature: Feature to read
            filters: Filter expressions
            **kwargs: Backend-specific options

        Returns:
            Narwhals LazyFrame or None if not found
        """
        ...

    @abstractmethod
    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop all metadata for a feature.

        Args:
            feature_key: Feature to drop
        """
        ...

    @abstractmethod
    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filters: Any,
        *,
        current_only: bool,
    ) -> None:
        """Delete metadata rows.

        Args:
            feature_key: Feature to delete from
            filters: Filter expressions
            current_only: If True, only delete non-deleted records (for soft deletes)
        """
        ...

    @abstractmethod
    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Check if feature exists in store.

        Args:
            feature: Feature to check

        Returns:
            True if feature table exists
        """
        ...
