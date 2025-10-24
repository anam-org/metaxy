"""DuckLake-specific data version calculator."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from metaxy.data_versioning.calculators.duckdb import (
    DuckDBDataVersionCalculator,
    ExtensionSpec,
)
from metaxy.data_versioning.calculators.ibis import HashSQLGenerator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    import ibis
    import narwhals as nw

    from metaxy.metadata_store.duckdb import ExtensionSpec
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan
else:
    from typing import Any

    ExtensionSpec = dict[str, Any]


class DuckLakeDataVersionCalculator(DuckDBDataVersionCalculator):
    """DuckLake calculator that ensures DuckLake plugins are installed."""

    def __init__(
        self,
        hash_sql_generators: dict[HashAlgorithm, HashSQLGenerator],
        *,
        alias: str = "ducklake",
        extensions: Iterable[ExtensionSpec | str] | None = None,
        connection: "ibis.BaseBackend | None" = None,
    ):
        ext_list = list(extensions or [])
        extension_names = {
            ext if isinstance(ext, str) else ext.get("name", "") for ext in ext_list
        }
        if "ducklake" not in extension_names:
            ext_list.append("ducklake")
            extension_names.add("ducklake")
        if "hashfuncs" not in extension_names:
            ext_list.append("hashfuncs")

        super().__init__(
            hash_sql_generators=hash_sql_generators,
            extensions=ext_list,
            connection=connection,
        )
        self.alias = alias
        self.extensions = ext_list
        self._hash_sql_generators = hash_sql_generators
        self._backend: ibis.BaseBackend | None = None
        self._calculator: IbisDataVersionCalculator | None = None

        if connection is not None:
            self.set_connection(connection)

    @property
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """Algorithms supported by configured hash SQL generators."""
        return list(self._hash_sql_generators.keys())

    @property
    def default_algorithm(self) -> HashAlgorithm:
        """Default algorithm prefers xxHash64 when available."""
        if HashAlgorithm.XXHASH64 in self.supported_algorithms:
            return HashAlgorithm.XXHASH64
        return self.supported_algorithms[0]

    def _ensure_ducklake_catalog(self) -> None:
        """Ensure subsequent queries run within the DuckLake catalog."""
        if self._backend is None:
            return

        try:
            self._backend.raw_sql(f"USE {self.alias}")  # type: ignore[attr-defined]
        except Exception:
            # Catalog already set or USE unsupported - proceed with existing context
            pass

    def set_connection(self, connection: "ibis.BaseBackend") -> None:
        super().set_connection(connection)
        # Ensure all queries operate within the DuckLake catalog
        try:
            connection.raw_sql(f"USE {self.alias}")  # type: ignore[attr-defined]
        except Exception:
            # Falling back to pre-configured catalog if USE fails (e.g., already set)
            pass
