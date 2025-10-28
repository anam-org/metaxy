"""DuckDB metadata store - thin wrapper around IbisMetadataStore."""

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store._ducklake_support import (
    DuckDBConnection,
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    DuckLakeConfigInput,
    build_ducklake_attachment,
    ensure_extensions_with_plugins,
)
from metaxy.metadata_store.ibis import IbisMetadataStore


class ExtensionSpec(TypedDict, total=False):
    """
    DuckDB extension specification.

    Can be expressed in TOML as:
        extensions = ["hashfuncs"]  # string form, uses 'community' repo
        extensions = [{name = "hashfuncs"}]  # dict form, uses 'community' repo
        extensions = [{name = "spatial", repository = "core_nightly"}]
        extensions = [{name = "my_ext", repository = "https://my-repo.com"}]
    """

    name: str
    repository: str  # defaults to "community" if not specified


class DuckDBMetadataStore(IbisMetadataStore):
    """
    DuckDB metadata store using Ibis backend.

    Convenience wrapper that configures IbisMetadataStore for DuckDB.

    Hash algorithm support is detected dynamically based on installed extensions:
    - MD5: Always available (built-in)
    - XXHASH32, XXHASH64: Available when 'hashfuncs' extension is loaded

    Components:
        - joiner: NarwhalsJoiner (works with any backend)
        - calculator: IbisDataVersionCalculator (native SQL hash computation with xxh64/xxh32/md5)
        - diff_resolver: NarwhalsDiffResolver

    Examples:
        >>> # Local file database
        >>> with DuckDBMetadataStore("metadata.db") as store:
        ...     store.write_metadata(MyFeature, df)

        >>> # In-memory database
        >>> with DuckDBMetadataStore(":memory:") as store:
        ...     store.write_metadata(MyFeature, df)

        >>> # MotherDuck
        >>> with DuckDBMetadataStore("md:my_database") as store:
        ...     store.write_metadata(MyFeature, df)

        >>> # With extensions
        >>> store = DuckDBMetadataStore(
        ...     "metadata.db",
        ...     hash_algorithm=HashAlgorithm.XXHASH64,
        ...     extensions=["hashfuncs"]
        ... )
        >>> with store:
        ...     store.write_metadata(MyFeature, df)
    """

    def __init__(
        self,
        database: str | Path,
        *,
        config: dict[str, str] | None = None,
        extensions: list[ExtensionSpec | str] | None = None,
        fallback_stores: list["MetadataStore"] | None = None,
        ducklake: DuckLakeConfigInput | None = None,
        **kwargs,
    ):
        """
        Initialize DuckDB metadata store.

        Args:
            database: Database connection string or path.
                - File path: "metadata.db" or Path("metadata.db")
                - In-memory: ":memory:"
                - MotherDuck: "md:my_database" or "md:my_database?motherduck_token=..."
                - S3: "s3://bucket/path/database.duckdb" (read-only via ATTACH)
                - HTTPS: "https://example.com/database.duckdb" (read-only via ATTACH)
                - Any valid DuckDB connection string

                Note: Parent directories are NOT created automatically. Ensure paths exist
                before initializing the store.
            config: Optional DuckDB configuration settings (e.g., {'threads': '4', 'memory_limit': '4GB'})
            extensions: List of DuckDB extensions to install and load on open.
                Can be strings (installed from 'community' repository) or dicts
                specifying both name and repository.

                Examples:
                    extensions=['hashfuncs']  # Install hashfuncs from community
                    extensions=[{'name': 'hashfuncs'}]  # Same as above
                    extensions=[{'name': 'spatial', 'repository': 'core_nightly'}]
                    extensions=[{'name': 'my_ext', 'repository': 'https://my-repo.com'}]
            fallback_stores: Ordered list of read-only fallback stores.
            ducklake: Optional DuckLake attachment configuration. Provide either a
                mapping with 'metadata_backend' and 'storage_backend' entries or a
                DuckLakeAttachmentConfig instance. When supplied, the DuckDB
                connection is configured to ATTACH the DuckLake catalog after open().
            **kwargs: Passed to IbisMetadataStore (e.g., hash_algorithm, graph)
        """
        database_str = str(database)

        # Build connection params for Ibis DuckDB backend
        # Ibis DuckDB backend accepts config params directly (not nested under 'config')
        connection_params = {"database": database_str}
        if config:
            connection_params.update(config)

        self.database = database_str
        base_extensions = list(extensions or [])

        self._ducklake_config: DuckLakeAttachmentConfig | None = None
        self._ducklake_attachment: DuckLakeAttachmentManager | None = None
        if ducklake is not None:
            attachment_config, manager = build_ducklake_attachment(ducklake)
            ensure_extensions_with_plugins(base_extensions, attachment_config.plugins)
            self._ducklake_config = attachment_config
            self._ducklake_attachment = manager

        self.extensions = base_extensions

        # Auto-add hashfuncs extension if not present (needed for default XXHASH64)
        extension_names = [
            ext if isinstance(ext, str) else ext.get("name", "")
            for ext in self.extensions
        ]
        if "hashfuncs" not in extension_names:
            self.extensions.append("hashfuncs")

        # Initialize Ibis store with DuckDB backend
        super().__init__(
            backend="duckdb",
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get default hash algorithm for DuckDB stores.

        Uses XXHASH64 which requires the hashfuncs extension (lazily loaded).
        """
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """DuckDB stores support native data version calculations when connection is open."""
        return self._conn is not None

    def _create_native_components(self):
        """Create components for native SQL execution with DuckDB.

        Uses DuckDBDataVersionCalculator which handles extension loading lazily.
        Extensions are loaded when the calculator is created (on-demand), not on store open.
        """
        from metaxy.data_versioning.calculators.duckdb import (
            DuckDBDataVersionCalculator,
        )
        from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
        from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner

        if self._conn is None:
            raise RuntimeError(
                "Cannot create native data version calculations: store is not open. "
                "Ensure store is used as context manager."
            )

        # All components accept/return Narwhals LazyFrames
        # DuckDBDataVersionCalculator loads extensions and generates SQL for hashing
        joiner = NarwhalsJoiner()
        calculator = DuckDBDataVersionCalculator(
            backend=self._conn,
            extensions=self.extensions,
        )
        diff_resolver = NarwhalsDiffResolver()

        return joiner, calculator, diff_resolver

    # ------------------------------------------------------------------ DuckLake
    def open(self) -> None:
        """Open DuckDB connection and configure optional DuckLake attachment."""
        super().open()
        if self._ducklake_attachment is not None:
            duckdb_conn = self._duckdb_raw_connection()
            self._ducklake_attachment.configure(duckdb_conn)

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

    def _duckdb_raw_connection(self) -> DuckDBConnection:
        """Return the underlying DuckDBPyConnection from the Ibis backend."""
        if self._conn is None:
            raise RuntimeError("DuckDB connection is not open.")

        backend = self._conn
        candidate = getattr(backend, "con", None)
        if isinstance(candidate, DuckDBConnection):
            return candidate

        raw_connection = getattr(backend, "raw_connection", None)
        if callable(raw_connection):
            raw_candidate = raw_connection()
            if isinstance(raw_candidate, DuckDBConnection):
                return raw_candidate

        raise RuntimeError(
            "DuckDB Ibis backend does not expose a DuckDBPyConnection via 'con' or raw_connection()."
        )
