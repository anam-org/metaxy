"""DuckDB metadata store - thin wrapper around IbisMetadataStore."""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from typing_extensions import Self

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy._decorators import public
from metaxy.ext.metadata_stores._ducklake_support import (
    DuckDBPyConnection,
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    MotherDuckMetadataBackendConfig,
)
from metaxy.metadata_store.ibis import IbisMetadataStore, IbisMetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.versioning.types import HashAlgorithm


@public
class ExtensionSpec(BaseModel):
    """DuckDB extension specification accepted by DuckDBMetadataStore."""

    name: str
    repository: str | None = None
    init_sql: Sequence[str] = ()
    """SQL statements to execute immediately after loading the extension."""


@public
class DuckDBMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for DuckDBMetadataStore.

    Example:
        ```python
        config = DuckDBMetadataStoreConfig(
            database="metadata.db",
            extensions=["hashfuncs"],
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

        store = DuckDBMetadataStore.from_config(config)
        ```
    """

    database: str | Path = Field(
        description="Database path (:memory:, file path, or md:database).",
    )
    config: dict[str, str] | None = Field(
        default=None,
        description="DuckDB configuration settings (e.g., {'threads': '4'}).",
    )
    extensions: Sequence[str | ExtensionSpec] | None = Field(
        default=None,
        description="DuckDB extensions to install and load on open.",
    )
    ducklake: DuckLakeAttachmentConfig | None = Field(
        default=None,
        description="DuckLake attachment configuration.",
    )


def _normalise_extensions(
    extensions: Iterable[str | ExtensionSpec],
) -> list[ExtensionSpec]:
    """Coerce extension inputs into ExtensionSpec instances."""
    normalised: list[ExtensionSpec] = []
    for ext in extensions:
        if isinstance(ext, str):
            normalised.append(ExtensionSpec(name=ext))
        elif isinstance(ext, ExtensionSpec):
            normalised.append(ext)
        else:
            raise TypeError(f"DuckDB extensions must be strings or ExtensionSpec instances, got {type(ext).__name__}.")
    return normalised


@public
class DuckDBMetadataStore(IbisMetadataStore):
    """
    [DuckDB](https://duckdb.org/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Example: Local File
        ```py
        store = DuckDBMetadataStore("metadata.db")
        ```

    Example: In-memory database
        ```py
        # In-memory database
        store = DuckDBMetadataStore(":memory:")
        ```

    Example: MotherDuck
        ```py
        # MotherDuck
        store = DuckDBMetadataStore("md:my_database")
        ```

    Example: With extensions
        ```py
        # With extensions
        store = DuckDBMetadataStore("metadata.db", hash_algorithm=HashAlgorithm.XXHASH64, extensions=["hashfuncs"])
        ```
    """

    def __init__(
        self,
        database: str | Path,
        *,
        config: dict[str, str] | None = None,
        extensions: Sequence[str | ExtensionSpec] | None = None,
        fallback_stores: list["MetadataStore"] | None = None,
        ducklake: DuckLakeAttachmentConfig | None = None,
        **kwargs,
    ):
        """
        Initialize [DuckDB](https://duckdb.org/) metadata store.

        Args:
            database: Database connection string or path.
                - File path: `"metadata.db"` or `Path("metadata.db")`

                - In-memory: `":memory:"`

                - MotherDuck: `"md:my_database"` or `"md:my_database?motherduck_token=..."`

                - S3: `"s3://bucket/path/database.duckdb"` (read-only via ATTACH)

                - HTTPS: `"https://example.com/database.duckdb"` (read-only via ATTACH)

                - Any valid DuckDB connection string

            config: Optional DuckDB configuration settings (e.g., {'threads': '4', 'memory_limit': '4GB'})
            extensions: List of DuckDB extensions to install and load on open.
                Supports strings (community repo) or
                [metaxy.ext.metadata_stores.duckdb.ExtensionSpec][] instances.
            ducklake: Optional [DuckLake](https://ducklake.select/) attachment configuration.
                When supplied, the DuckDB connection is configured to ATTACH the
                DuckLake catalog after open().
            fallback_stores: Ordered list of read-only fallback stores.

        Warning:
            Parent directories are NOT created automatically. Ensure paths exist
            before initializing the store.
        """
        database_str = str(database)

        connection_params = {"database": database_str}
        if config:
            connection_params.update(config)

        self.database = database_str
        self.extensions: list[ExtensionSpec] = _normalise_extensions(extensions or [])

        self._ducklake_config: DuckLakeAttachmentConfig | None = None
        self._ducklake_attachment: DuckLakeAttachmentManager | None = None
        if ducklake is not None:
            existing_names = {ext.name for ext in self.extensions}
            if "ducklake" not in existing_names:
                self.extensions.append(ExtensionSpec(name="ducklake"))
            if (
                isinstance(ducklake.metadata_backend, MotherDuckMetadataBackendConfig)
                and "motherduck" not in existing_names
            ):
                self.extensions.append(ExtensionSpec(name="motherduck"))
            self._ducklake_config = ducklake
            self._ducklake_attachment = DuckLakeAttachmentManager(ducklake, store_name=kwargs.get("name"))

        if "hashfuncs" not in {ext.name for ext in self.extensions}:
            self.extensions.append(ExtensionSpec(name="hashfuncs", repository="community"))

        super().__init__(
            backend="duckdb",
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    @property
    def sqlalchemy_url(self) -> str:
        """Get SQLAlchemy-compatible connection URL for DuckDB.

        Constructs a DuckDB SQLAlchemy URL from the database parameter.

        Returns:
            SQLAlchemy-compatible URL string (e.g., "duckdb:///path/to/db.db")

        Example:
            ```python
            store = DuckDBMetadataStore(":memory:")
            print(store.sqlalchemy_url)  # duckdb:///:memory:

            store = DuckDBMetadataStore("metadata.db")
            print(store.sqlalchemy_url)  # duckdb:///metadata.db
            ```
        """
        # DuckDB SQLAlchemy URL format: duckdb:///database_path
        return f"duckdb:///{self.database}"

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for DuckDB stores.

        Uses XXHASH32 which requires the hashfuncs extension.
        """
        return HashAlgorithm.XXHASH32

    @contextmanager
    def _create_versioning_engine(self, plan):
        """Create provenance engine for DuckDB backend as a context manager.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            IbisVersioningEngine with DuckDB-specific hash functions.
        """
        with super()._create_versioning_engine(plan) as engine:
            yield engine

    def _load_extensions(self) -> None:
        """Install and load all configured DuckDB extensions."""
        if not self.extensions:
            return

        duckdb_conn = self._duckdb_raw_connection()
        for ext in self.extensions:
            if ext.repository is not None:
                duckdb_conn.install_extension(ext.name, repository=ext.repository)
            else:
                duckdb_conn.install_extension(ext.name)
            duckdb_conn.load_extension(ext.name)
            for sql in ext.init_sql:
                duckdb_conn.execute(sql)

    def _create_hash_functions(self):
        """Create DuckDB-specific hash functions for Ibis expressions.

        Implements MD5 and xxHash functions using DuckDB's native functions.

        Returns hash functions that take Ibis column expressions and return
        Ibis expressions that call DuckDB SQL functions.
        """
        # Import ibis for wrapping built-in SQL functions
        import ibis

        hash_functions = {}

        # DuckDB MD5 implementation
        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:  # ty: ignore[empty-body]
            """DuckDB MD5() function."""
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:  # ty: ignore[empty-body]
            """DuckDB HEX() function."""
            ...

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str:  # ty: ignore[empty-body]
            """DuckDB LOWER() function."""
            ...

        def md5_hash(col_expr):
            """Hash a column using DuckDB's MD5() function."""
            # MD5 already returns hex string, just convert to lowercase
            return LOWER(MD5(col_expr.cast(str)))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        # Add xxHash functions if hashfuncs extension is loaded
        if "hashfuncs" in {ext.name for ext in self.extensions}:
            # Use Ibis's builtin UDF decorator to wrap DuckDB's xxhash functions
            # These functions already exist in DuckDB (via hashfuncs extension)
            # The decorator tells Ibis to call them directly in SQL
            # NOTE: xxh32/xxh64 return integers in DuckDB, not strings
            @ibis.udf.scalar.builtin
            def xxh32(x: str) -> int:  # ty: ignore[empty-body]
                """DuckDB xxh32() hash function from hashfuncs extension."""
                ...

            @ibis.udf.scalar.builtin
            def xxh64(x: str) -> int:  # ty: ignore[empty-body]
                """DuckDB xxh64() hash function from hashfuncs extension."""
                ...

            # Create hash functions that use these wrapped SQL functions
            def xxhash32_hash(col_expr):
                """Hash a column using DuckDB's xxh32() function."""
                # Cast to string and then cast result to string (xxh32 returns integer in DuckDB)
                return xxh32(col_expr.cast(str)).cast(str)

            def xxhash64_hash(col_expr):
                """Hash a column using DuckDB's xxh64() function."""
                # Cast to string and then cast result to string (xxh64 returns integer in DuckDB)
                return xxh64(col_expr.cast(str)).cast(str)

            hash_functions[HashAlgorithm.XXHASH32] = xxhash32_hash
            hash_functions[HashAlgorithm.XXHASH64] = xxhash64_hash

        return hash_functions

    # ------------------------------------------------------------------ DuckLake
    @contextmanager
    def open(self, mode: AccessMode = "r") -> Iterator[Self]:
        """Open DuckDB connection with specified access mode.

        Args:
            mode: Access mode (READ or WRITE). Defaults to READ.
                READ mode sets read_only=True for concurrent access.

        Yields:
            Self: The store instance with connection open
        """
        # Setup: Configure connection params based on mode
        if mode == "r":
            self.connection_params["read_only"] = True
        else:
            # Remove read_only if present (switching to WRITE)
            self.connection_params.pop("read_only", None)

        # Call parent context manager to establish connection
        # Each outermost open() creates a fresh DuckDB connection (see
        # IbisMetadataStore.open).  Extensions and DuckLake attachment are
        # per-connection state, so they must be configured on every entry at
        # depth 1.  install_extension() is a no-op when already on disk;
        # load_extension() is required per-connection.
        with super().open(mode):
            if self._context_depth == 1:
                self._load_extensions()
                if self._ducklake_attachment is not None:
                    self._ducklake_attachment._attached = False
                    self._ducklake_attachment.configure(self._duckdb_raw_connection())

            yield self

    def preview_ducklake_sql(self) -> list[str]:
        """Return DuckLake attachment SQL if configured."""
        return self.ducklake_attachment.preview_sql()

    @property
    def ducklake_attachment(self) -> DuckLakeAttachmentManager:
        """DuckLake attachment manager (raises if not configured)."""
        if self._ducklake_attachment is None:
            raise RuntimeError("DuckLake attachment is not configured.")
        return self._ducklake_attachment

    @property
    def ducklake_attachment_config(self) -> DuckLakeAttachmentConfig:
        """DuckLake attachment configuration (raises if not configured)."""
        if self._ducklake_config is None:
            raise RuntimeError("DuckLake attachment is not configured.")
        return self._ducklake_config

    def _duckdb_raw_connection(self) -> DuckDBPyConnection:
        """Return the underlying DuckDBPyConnection from the Ibis backend."""
        if self._conn is None:
            raise RuntimeError("DuckDB connection is not open.")

        candidate = self._conn.con  # ty: ignore[unresolved-attribute]

        if not isinstance(candidate, DuckDBPyConnection):
            raise TypeError(f"Expected DuckDB backend 'con' to be DuckDBPyConnection, got {type(candidate).__name__}")

        return candidate

    @classmethod
    def from_config(cls, config: DuckDBMetadataStoreConfig, **kwargs: Any) -> Self:  # type: ignore[override]
        from metaxy.config import MetaxyConfig
        from metaxy.metadata_store.fallback import FallbackStoreList

        config_dict = config.model_dump(exclude_unset=True, exclude={"ducklake", "fallback_stores"})
        store = cls(ducklake=config.ducklake, **config_dict, **kwargs)
        fallback_store_names = config.model_dump(exclude_unset=True).get("fallback_stores", [])
        if fallback_store_names:
            store.fallback_stores = FallbackStoreList(
                fallback_store_names,
                config=MetaxyConfig.get(),
                parent_hash_algorithm=store.hash_algorithm,
            )
        return store

    @classmethod
    def config_model(cls) -> type[DuckDBMetadataStoreConfig]:
        return DuckDBMetadataStoreConfig
