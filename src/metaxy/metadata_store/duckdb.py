"""DuckDB metadata store - thin wrapper around IbisMetadataStore."""

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
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
            **kwargs: Passed to IbisMetadataStore (e.g., hash_algorithm, graph)
        """
        database_str = str(database)

        # Build connection params for Ibis DuckDB backend
        # Ibis DuckDB backend accepts config params directly (not nested under 'config')
        connection_params = {"database": database_str}
        if config:
            connection_params.update(config)

        self.database = database_str
        self.extensions = extensions or []

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

        Uses XXHASH64 which requires the hashfuncs extension (auto-loaded).
        """
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """DuckDB stores support native components when connection is open."""
        return self._conn is not None

    def _get_hash_sql_generators(self) -> dict[HashAlgorithm, "HashSQLGenerator"]:
        """Get hash SQL generators for DuckDB.

        DuckDB supports:
        - MD5: Always available (built-in)
        - XXHASH32, XXHASH64: Available when 'hashfuncs' extension is loaded

        Returns:
            Dictionary mapping HashAlgorithm to SQL generator functions
        """
        generators: dict[HashAlgorithm, HashSQLGenerator] = {}

        # MD5 is always available
        def md5_generator(table, concat_columns: dict[str, str]) -> str:
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                # Cast to VARCHAR for consistency
                hash_expr = f"CAST(md5({concat_col}) AS VARCHAR)"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        generators[HashAlgorithm.MD5] = md5_generator

        # Check if hashfuncs extension is in the list
        extension_names = [
            ext if isinstance(ext, str) else ext.get("name", "")
            for ext in self.extensions
        ]

        if "hashfuncs" in extension_names:

            def xxhash32_generator(table, concat_columns: dict[str, str]) -> str:
                hash_selects: list[str] = []
                for field_key, concat_col in concat_columns.items():
                    hash_col = f"__hash_{field_key}"
                    hash_expr = f"CAST(xxh32({concat_col}) AS VARCHAR)"
                    hash_selects.append(f"{hash_expr} as {hash_col}")

                hash_clause = ", ".join(hash_selects)
                table_sql = table.compile()
                return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

            def xxhash64_generator(table, concat_columns: dict[str, str]) -> str:
                hash_selects: list[str] = []
                for field_key, concat_col in concat_columns.items():
                    hash_col = f"__hash_{field_key}"
                    hash_expr = f"CAST(xxh64({concat_col}) AS VARCHAR)"
                    hash_selects.append(f"{hash_expr} as {hash_col}")

                hash_clause = ", ".join(hash_selects)
                table_sql = table.compile()
                return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

            generators[HashAlgorithm.XXHASH32] = xxhash32_generator
            generators[HashAlgorithm.XXHASH64] = xxhash64_generator

        return generators
