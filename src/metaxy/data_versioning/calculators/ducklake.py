"""DuckLake-specific data version calculator."""

from typing import TYPE_CHECKING, Iterable

from metaxy.data_versioning.calculators.base import DataVersionCalculator
from metaxy.data_versioning.calculators.ibis import (
    HashSQLGenerator,
    IbisDataVersionCalculator,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    import ibis
    import narwhals as nw

    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan
    from metaxy.metadata_store.duckdb import ExtensionSpec
else:
    from typing import Any

    ExtensionSpec = dict[str, Any]


class DuckLakeDataVersionCalculator(DataVersionCalculator):
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

        self.alias = alias
        self.extensions = ext_list
        self._hash_sql_generators = hash_sql_generators
        self._backend: "ibis.BaseBackend | None" = None
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
        """Attach an Ibis backend connection and configure DuckLake context."""
        self._backend = connection
        self._calculator = IbisDataVersionCalculator(
            backend=connection,
            hash_sql_generators=self._hash_sql_generators,
        )
        self._ensure_ducklake_catalog()

    def calculate_data_versions(
        self,
        joined_upstream: "nw.LazyFrame",
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> "nw.LazyFrame":
        """Delegate hashing to the wrapped Ibis calculator within DuckLake catalog."""
        if self._calculator is None:
            raise RuntimeError(
                "DuckLakeDataVersionCalculator requires an active Ibis backend. "
                "Call set_connection() before calculating data versions."
            )

        self._ensure_ducklake_catalog()

        return self._calculator.calculate_data_versions(
            joined_upstream=joined_upstream,
            feature_spec=feature_spec,
            feature_plan=feature_plan,
            upstream_column_mapping=upstream_column_mapping,
            hash_algorithm=hash_algorithm,
        )
