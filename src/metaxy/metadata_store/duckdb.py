"""DuckDB metadata store - thin wrapper around IbisMetadataStore."""

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy.metadata_store._ducklake_support import (
    DuckDBPyConnection,
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    DuckLakeConfigInput,
    build_ducklake_attachment,
    ensure_extensions_with_plugins,
)
from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.provenance.types import HashAlgorithm


class ExtensionSpec(BaseModel):
    """
    DuckDB extension specification accepted by DuckDBMetadataStore.

    Supports additional keys for forward compatibility.
    """

    name: str
    repository: str | None = None

    model_config = ConfigDict(extra="allow")


ExtensionInput = str | ExtensionSpec | Mapping[str, Any]
NormalisedExtension = str | ExtensionSpec


def _normalise_extensions(
    extensions: Iterable[ExtensionInput],
) -> list[NormalisedExtension]:
    """Coerce extension inputs into strings or fully-validated specs."""
    normalised: list[NormalisedExtension] = []
    for ext in extensions:
        if isinstance(ext, str):
            normalised.append(ext)
        elif isinstance(ext, ExtensionSpec):
            normalised.append(ext)
        elif isinstance(ext, Mapping):
            try:
                normalised.append(ExtensionSpec.model_validate(ext))
            except ValidationError as exc:
                raise ValueError(f"Invalid DuckDB extension spec: {ext!r}") from exc
        else:
            raise TypeError(
                "DuckDB extensions must be strings or mapping-like objects with a 'name'."
            )
    return normalised


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
        store = DuckDBMetadataStore(
            "metadata.db",
            hash_algorithm=HashAlgorithm.XXHASH64,
            extensions=["hashfuncs"]
        )
        ```
    """

    def __init__(
        self,
        database: str | Path,
        *,
        config: dict[str, str] | None = None,
        extensions: Sequence[ExtensionInput] | None = None,
        fallback_stores: list["MetadataStore"] | None = None,
        ducklake: DuckLakeConfigInput | None = None,
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
                Supports strings (community repo), mapping-like objects with
                ``name``/``repository`` keys, or [metaxy.metadata_store.duckdb.ExtensionSpec][] instances.

        ducklake: Optional DuckLake attachment configuration. Provide either a
            mapping with 'metadata_backend' and 'storage_backend' entries or a
            DuckLakeAttachmentConfig instance. When supplied, the DuckDB
            connection is configured to ATTACH the DuckLake catalog after open().
            fallback_stores: Ordered list of read-only fallback stores.

            **kwargs: Passed to [metaxy.metadata_store.ibis.IbisMetadataStore][]`

        Warning:
            Parent directories are NOT created automatically. Ensure paths exist
            before initializing the store.
        """
        database_str = str(database)

        # Build connection params for Ibis DuckDB backend
        # Ibis DuckDB backend accepts config params directly (not nested under 'config')
        connection_params = {"database": database_str}
        if config:
            connection_params.update(config)

        self.database = database_str
        base_extensions: list[NormalisedExtension] = _normalise_extensions(
            extensions or []
        )

        self._ducklake_config: DuckLakeAttachmentConfig | None = None
        self._ducklake_attachment: DuckLakeAttachmentManager | None = None
        if ducklake is not None:
            attachment_config, manager = build_ducklake_attachment(ducklake)
            ensure_extensions_with_plugins(base_extensions, attachment_config.plugins)
            self._ducklake_config = attachment_config
            self._ducklake_attachment = manager

        self.extensions = base_extensions

        # Auto-add hashfuncs extension if not present (needed for default XXHASH64)
        extension_names: list[str] = []
        for ext in self.extensions:
            if isinstance(ext, str):
                extension_names.append(ext)
            elif isinstance(ext, ExtensionSpec):
                extension_names.append(ext.name)
            else:
                # After _normalise_extensions, this should not happen
                # But keep defensive check for type safety
                raise TypeError(
                    f"Extension must be str or ExtensionSpec after normalization; got {type(ext)}"
                )
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
        """DuckDB stores support native provenance tracking when connection is open."""
        return self._conn is not None

    @contextmanager
    def _create_provenance_tracker(self, plan):
        """Create provenance tracker for DuckDB backend as a context manager.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            IbisProvenanceTracker with DuckDB-specific hash functions.

        Note:
            Extensions are loaded lazily when tracker is created.
        """
        # Load extensions first (if connection is open)
        if self._conn is not None:
            self._load_extensions()

        # Call parent implementation (which calls our _create_hash_functions)
        with super()._create_provenance_tracker(plan) as tracker:
            yield tracker

    def _load_extensions(self) -> None:
        """Load DuckDB extensions if not already loaded."""
        if not self.extensions:
            return

        # Get raw DuckDB connection
        duckdb_conn = self._duckdb_raw_connection()

        for ext_spec in self.extensions:
            # Extract name and repository
            if isinstance(ext_spec, str):
                ext_name = ext_spec
                ext_repo = "community"
            elif isinstance(ext_spec, ExtensionSpec):
                ext_name = ext_spec.name
                ext_repo = ext_spec.repository or "community"
            else:
                raise TypeError(
                    f"Extension must be str or ExtensionSpec; got {type(ext_spec)}"
                )

            # Install and load the extension
            if ext_repo == "community":
                duckdb_conn.execute(f"INSTALL {ext_name} FROM community")
            else:
                duckdb_conn.execute(f"SET custom_extension_repository='{ext_repo}'")
                duckdb_conn.execute(f"INSTALL {ext_name}")

            duckdb_conn.execute(f"LOAD {ext_name}")

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
        def MD5(x: str) -> str:
            """DuckDB MD5() function."""
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:
            """DuckDB HEX() function."""
            ...

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str:
            """DuckDB LOWER() function."""
            ...

        def md5_hash(col_expr):
            """Hash a column using DuckDB's MD5() function."""
            # MD5 already returns hex string, just convert to lowercase
            return LOWER(MD5(col_expr.cast(str)))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        # Determine which extensions are available
        extension_names = []
        for ext in self.extensions:
            if isinstance(ext, str):
                extension_names.append(ext)
            elif isinstance(ext, ExtensionSpec):
                extension_names.append(ext.name)

        # Add xxHash functions if hashfuncs extension is loaded
        if "hashfuncs" in extension_names:
            # Use Ibis's builtin UDF decorator to wrap DuckDB's xxhash functions
            # These functions already exist in DuckDB (via hashfuncs extension)
            # The decorator tells Ibis to call them directly in SQL
            # NOTE: xxh32/xxh64 return integers in DuckDB, not strings
            @ibis.udf.scalar.builtin
            def xxh32(x: str) -> int:
                """DuckDB xxh32() hash function from hashfuncs extension."""
                ...

            @ibis.udf.scalar.builtin
            def xxh64(x: str) -> int:
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
    def open(self) -> None:
        """Open DuckDB connection and configure optional DuckLake attachment."""
        super().open()
        if self._ducklake_attachment is not None:
            try:
                duckdb_conn = self._duckdb_raw_connection()
                self._ducklake_attachment.configure(duckdb_conn)
            except Exception:
                # Ensure connection is closed if DuckLake configuration fails
                super().close()
                raise

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

        candidate = self._conn.con  # pyright: ignore[reportAttributeAccessIssue]

        if not isinstance(candidate, DuckDBPyConnection):
            raise TypeError(
                f"Expected DuckDB backend 'con' to be DuckDBPyConnection, "
                f"got {type(candidate).__name__}"
            )

        return candidate
