"""DuckDB-specific data version calculator with extension management."""

from typing import TYPE_CHECKING, TypedDict

from metaxy.data_versioning.calculators.ibis import (
    HashSQLGenerator,
    IbisDataVersionCalculator,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    import ibis


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


class DuckDBDataVersionCalculator(IbisDataVersionCalculator):
    """
    DuckDB-specific data version calculator that handles extension installation.

    Installs extensions lazily only when actually computing data versions,
    not when the store opens. This avoids unnecessary overhead when creating
    many store instances.
    """

    def __init__(
        self,
        hash_sql_generators: dict[HashAlgorithm, HashSQLGenerator],
        extensions: list[ExtensionSpec | str] | None = None,
        connection: "ibis.BaseBackend | None" = None,
    ):
        """
        Initialize DuckDB data version calculator.

        Args:
            hash_sql_generators: Map from HashAlgorithm to SQL generator function
            extensions: List of DuckDB extensions to install
            connection: Optional Ibis connection (can be set later)
        """
        super().__init__(hash_sql_generators)
        self.extensions = extensions or ["hashfuncs"]
        self._connection = connection
        self._extensions_installed = False

    @property
    def default_algorithm(self) -> HashAlgorithm:
        """DuckDB always uses XXHASH64 as default."""
        return HashAlgorithm.XXHASH64

    def set_connection(self, connection: "ibis.BaseBackend") -> None:
        """
        Set the database connection.

        Args:
            connection: Ibis connection to DuckDB
        """
        self._connection = connection

    def _ensure_extensions_installed(self) -> None:
        """
        Lazily install extensions when first needed.

        Called automatically before calculating data versions.
        """
        if self._extensions_installed or not self._connection:
            return

        for ext in self.extensions:
            if isinstance(ext, str):
                name = ext
                repository = "community"
            else:
                name = ext["name"]
                repository = ext.get("repository", "community")

            self._connection.raw_sql(f"INSTALL {name} FROM {repository}")  # type: ignore[attr-defined]
            self._connection.raw_sql(f"LOAD {name}")  # type: ignore[attr-defined]

        self._extensions_installed = True

    def calculate_data_versions(self, *args, **kwargs):
        """
        Calculate data versions, installing extensions if needed.

        Overrides parent to ensure extensions are installed before computation.
        """
        # Install extensions lazily on first use
        self._ensure_extensions_installed()

        # Delegate to parent implementation
        return super().calculate_data_versions(*args, **kwargs)
