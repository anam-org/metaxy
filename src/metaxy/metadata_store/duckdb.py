"""DuckDB metadata store - thin wrapper around IbisMetadataStore."""

from pathlib import Path
from typing import TYPE_CHECKING

import ibis.expr.types as ir

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.calculators.duckdb import (
    DuckDBDataVersionCalculator,
    ExtensionSpec,
)
from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
from metaxy.data_versioning.diff.ibis import IbisDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.ibis import IbisJoiner
from metaxy.metadata_store.ibis import IbisMetadataStore


class DuckDBMetadataStore(IbisMetadataStore):
    """
    DuckDB metadata store using Ibis backend.

    Convenience wrapper that configures IbisMetadataStore for DuckDB.

    Hash algorithm support is detected dynamically based on installed extensions:
    - MD5: Always available (built-in)
    - XXHASH32, XXHASH64: Available when 'hashfuncs' extension is loaded

    Components:
        - joiner: IbisJoiner | PolarsJoiner (based on prefer_native)
        - calculator: DuckDBDataVersionCalculator | PolarsDataVersionCalculator
        - diff_resolver: IbisDiffResolver | PolarsDiffResolver

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
            **kwargs: Passed to IbisMetadataStore (e.g., hash_algorithm, registry)
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

    def _create_native_components(self):
        """Create DuckDB-specific native components."""
        from metaxy.data_versioning.calculators.base import DataVersionCalculator
        from metaxy.data_versioning.diff.base import MetadataDiffResolver
        from metaxy.data_versioning.joiners.base import UpstreamJoiner

        if self._conn is None:
            raise RuntimeError(
                "Cannot create native components: store is not open. "
                "Ensure store is used as context manager."
            )

        import ibis.expr.types as ir

        joiner: UpstreamJoiner[ir.Table] = IbisJoiner(backend=self._conn)
        calculator: DataVersionCalculator[ir.Table] = DuckDBDataVersionCalculator(
            hash_sql_generators=self.hash_sql_generators,
            extensions=self.extensions,
        )
        # Give calculator access to connection for lazy extension installation
        calculator.set_connection(self._conn)  # type: ignore[attr-defined]

        diff_resolver: MetadataDiffResolver[ir.Table] = IbisDiffResolver()

        return joiner, calculator, diff_resolver

    @property
    def hash_sql_generators(self) -> dict[HashAlgorithm, HashSQLGenerator]:
        """
        Build hash SQL generators based on declared extensions.

        Returns generators for algorithms that will be available based on
        the extensions list. Does not query the database.

        Returns:
            Dictionary mapping HashAlgorithm to SQL generator functions
        """
        generators = {HashAlgorithm.MD5: self._generate_hash_sql(HashAlgorithm.MD5)}

        # Check if hashfuncs is in extensions list (static check, no DB access)
        extension_names = [
            ext if isinstance(ext, str) else ext.get("name", "")
            for ext in self.extensions
        ]

        if "hashfuncs" in extension_names:
            generators[HashAlgorithm.XXHASH32] = self._generate_hash_sql(
                HashAlgorithm.XXHASH32
            )
            generators[HashAlgorithm.XXHASH64] = self._generate_hash_sql(
                HashAlgorithm.XXHASH64
            )

        return generators

    def _generate_hash_sql(self, algorithm: HashAlgorithm) -> HashSQLGenerator:
        """
        Generate SQL hash function for DuckDB.

        Creates a hash SQL generator that produces DuckDB-specific SQL for
        computing hash values on concatenated columns.

        Args:
            algorithm: Hash algorithm to generate SQL for

        Returns:
            Hash SQL generator function

        Raises:
            ValueError: If algorithm is not supported by DuckDB
        """
        # Map algorithm to DuckDB function name
        if algorithm == HashAlgorithm.MD5:
            hash_function = "md5"
        elif algorithm == HashAlgorithm.XXHASH32:
            hash_function = "xxh32"
        elif algorithm == HashAlgorithm.XXHASH64:
            hash_function = "xxh64"
        else:
            from metaxy.metadata_store.exceptions import (
                HashAlgorithmNotSupportedError,
            )

            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm {algorithm} is not supported by DuckDB. "
                f"Supported algorithms: MD5 (built-in), XXHASH32, XXHASH64 (require 'hashfuncs' extension)"
            )

        def generator(table: ir.Table, concat_columns: dict[str, str]) -> str:
            # Build SELECT clause with hash columns
            hash_selects: list[str] = []
            for field_key, concat_col in concat_columns.items():
                hash_col = f"__hash_{field_key}"
                # Always cast to VARCHAR for consistency
                hash_expr = f"CAST({hash_function}({concat_col}) AS VARCHAR)"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        return generator

    def open(self) -> None:
        """Open DuckDB connection."""
        # Establish connection using parent class implementation
        super().open()
