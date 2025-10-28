"""DuckDB-specific data version calculator with extension management.

This calculator extends IbisDataVersionCalculator to handle DuckDB-specific
extension loading (e.g., hashfuncs for xxHash support).
"""
# pyright: reportImportCycles=false

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from metaxy.data_versioning.calculators.ibis import IbisDataVersionCalculator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    import ibis

    from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
    from metaxy.metadata_store.duckdb import ExtensionInput


class DuckDBDataVersionCalculator(IbisDataVersionCalculator):
    """DuckDB-specific calculator that manages extensions lazily.

    This calculator:
    1. Installs and loads DuckDB extensions on first use (lazy loading)
    2. Supports xxHash64, xxHash32, and MD5 hash functions
    3. Generates DuckDB-specific SQL for hash computation

    The extension loading happens in __init__, which is only called when
    native data version calculations are actually needed (not on store open).

    Example:
        >>> backend = ibis.duckdb.connect("metadata.db")
        >>> calculator = DuckDBDataVersionCalculator(
        ...     backend=backend,
        ...     extensions=["hashfuncs"]
        ... )
        >>> # Extensions are now loaded and xxHash64 is available
    """

    def __init__(
        self,
        backend: "ibis.BaseBackend",
        extensions: "Sequence[ExtensionInput] | None" = None,
    ):
        """Initialize DuckDB calculator and load extensions.

        Args:
            backend: DuckDB Ibis backend connection
            extensions: List of DuckDB extensions to install/load.
                Can be strings (from 'community' repo) or dicts with
                'name' and optional 'repository' keys.

        Example:
            >>> extensions = ["hashfuncs"]  # Simple form
            >>> extensions = [{"name": "spatial", "repository": "core_nightly"}]
        """
        self._backend = backend
        self.extensions = list(extensions or [])

        # Load extensions immediately (lazy at calculator creation time)
        self._load_extensions()

        # Generate hash SQL generators for DuckDB
        hash_sql_generators = self._generate_hash_sql_generators()

        # Initialize parent with backend and generators
        super().__init__(
            backend=backend,
            hash_sql_generators=hash_sql_generators,
        )

    def _load_extensions(self) -> None:
        """Install and load DuckDB extensions.

        This is called once when the calculator is created, which happens
        lazily when native data version calculations are first needed.
        """
        if not self.extensions:
            return

        # Type narrowing: we know this is a DuckDB backend
        from typing import cast

        backend = cast(
            Any, self._backend
        )  # DuckDB backend has raw_sql but not in ibis.BaseBackend stubs

        for ext_spec in self.extensions:
            if isinstance(ext_spec, str):
                ext_name = ext_spec
                ext_repo = "community"
            elif isinstance(ext_spec, Mapping):
                ext_name = str(ext_spec.get("name", ""))
                ext_repo = str(ext_spec.get("repository", "community"))
            else:
                ext_name = str(getattr(ext_spec, "name", ""))
                ext_repo = getattr(ext_spec, "repository", None) or "community"

            if not ext_name:
                raise ValueError("DuckDB extension specification must include a name.")

            if ext_repo != "community":
                backend.raw_sql(f"SET custom_extension_repository='{ext_repo}'")

            backend.raw_sql(f"INSTALL {ext_name}")
            backend.raw_sql(f"LOAD {ext_name}")

    def _generate_hash_sql_generators(self) -> dict[HashAlgorithm, "HashSQLGenerator"]:
        """Generate hash SQL generators for DuckDB.

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
                # md5() in DuckDB returns hex string, cast to VARCHAR for consistency
                hash_expr = f"CAST(md5({concat_col}) AS VARCHAR)"
                hash_selects.append(f"{hash_expr} as {hash_col}")

            hash_clause = ", ".join(hash_selects)
            table_sql = table.compile()
            return f"SELECT *, {hash_clause} FROM ({table_sql}) AS __metaxy_temp"

        generators[HashAlgorithm.MD5] = md5_generator

        # Check if hashfuncs extension is in the list
        extension_names = []
        for ext in self.extensions:
            if isinstance(ext, str):
                extension_names.append(ext)
            elif isinstance(ext, Mapping):
                extension_names.append(str(ext.get("name", "")))
            else:
                extension_names.append(str(getattr(ext, "name", "")))

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

    @property
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """Algorithms supported by this calculator based on loaded extensions."""
        # Dynamically determine based on what was actually loaded
        return list(self._hash_sql_generators.keys())

    @property
    def default_algorithm(self) -> HashAlgorithm:
        """Default hash algorithm for DuckDB.

        Uses XXHASH64 if hashfuncs extension is loaded, otherwise MD5.
        """
        if HashAlgorithm.XXHASH64 in self.supported_algorithms:
            return HashAlgorithm.XXHASH64
        return HashAlgorithm.MD5
