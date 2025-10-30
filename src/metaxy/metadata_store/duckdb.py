"""DuckDB metadata store - thin wrapper around IbisMetadataStore."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.metadata_store._ducklake_support import (
    DuckDBPyConnection,
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    DuckLakeConfigInput,
    build_ducklake_attachment,
    ensure_extensions_with_plugins,
)
from metaxy.metadata_store.ibis import IbisMetadataStore


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
        extensions: Sequence[ExtensionInput] | None = None,
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
                Supports strings (community repo), mapping-like objects with
                ``name``/``repository`` keys, or ExtensionSpec instances.

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
