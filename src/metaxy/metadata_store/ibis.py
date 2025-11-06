from abc import ABC
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import narwhals as nw
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.provenance.ibis import IbisHashFn, IbisProvenanceTracker
from metaxy.provenance.tracker import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


class IbisMetadataStore(MetadataStore, ABC):
    def __init__(
        self,
        hash_functions: dict[HashAlgorithm, IbisHashFn],
        hash_algo: HashAlgorithm,
        hash_length: int,
        auto_create_tables: bool = True,
    ):
        super().__init__(hash_algo, hash_length, auto_create_tables)

        self.hash_functions = hash_functions

        self._conn: Any = None  # Type will be ibis.backends.sql.SQLBackend when open

    @contextmanager
    def _open_with_connection(self, connection: Any) -> Iterator[Self]:
        """Internal method to open store with an Ibis connection.

        This is used by subclasses to provide a connection to the Ibis backend.
        Subclasses should override open() and call this method.

        Args:
            connection: Ibis backend connection

        Yields:
            Self: The store instance
        """
        try:
            self._conn = connection
            yield self
        finally:
            self._conn = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    @property
    def conn(self) -> Any:  # Returns ibis.backends.sql.SQLBackend
        from metaxy.metadata_store.exceptions import StoreNotOpenError
        if not self.is_open:
            raise StoreNotOpenError("Ibis connection is not opened.")
        assert self._conn is not None
        return self._conn

    @property
    def backend(self) -> str:
        return self.conn.name

    @property
    def hash_algorithm(self) -> HashAlgorithm:
        return self.hash_algo

    def close(self) -> None:
        self._conn = None

    def supports_native_tracker(self) -> bool:
        return True

    def create_tracker(self, plan: FeaturePlan) -> ProvenanceTracker:
        return IbisProvenanceTracker(plan, self.hash_functions)

    def read_metadata_impl(
        self,
        feature: type[BaseFeature] | FeatureKey,
        filters: Sequence[nw.Expr] | None = None,
    ) -> Frame | None:

        if isinstance(feature, FeatureKey):
            feature_key = feature
        else:
            feature_key = feature.spec().key
        table_name = feature_key.table_name

        if table_name not in self.conn.list_tables():
            return None  # type: ignore[return-value]

        table = self.conn.table(table_name)
        nw_lazy = nw.from_native(table)

        # Apply filters
        if filters:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        return nw_lazy

    def write_metadata(
        self, feature: type[BaseFeature] | FeatureKey, data: Frame
    ) -> None:
        if isinstance(feature, FeatureKey):
            feature_key = feature
        else:
            feature_key = feature.spec().key
        table_name = feature_key.table_name

        # Convert to Narwhals if needed
        nw_data = nw.from_native(data) if not isinstance(data, nw.DataFrame | nw.LazyFrame) else data  # type: ignore[arg-type]

        # Import ibis locally to avoid module-level import
        import ibis

        # If data is Polars, use DuckDB's native Polars support
        if nw_data.implementation == nw.Implementation.POLARS:
            # Get the Polars DataFrame
            polars_df = nw_data.to_polars()

            # Collect if lazy
            if hasattr(polars_df, 'collect'):
                polars_df = polars_df.collect()

            # Get the underlying DuckDB connection and register the Polars DataFrame
            # DuckDB can work directly with Polars via Arrow
            import ibis.backends.duckdb
            duckdb_backend = cast(ibis.backends.duckdb.Backend, self.conn)
            duckdb_conn = duckdb_backend.con

            # Register the Polars DataFrame so DuckDB can query it
            temp_view = f"__temp_{table_name}"
            duckdb_conn.register(temp_view, polars_df)

            try:
                if table_name not in self.conn.list_tables():
                    if self.auto_create_tables:
                        duckdb_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {temp_view}")
                    else:
                        raise ValueError(f"Table '{table_name}' does not exist and auto_create_tables is False")
                else:
                    duckdb_conn.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_view}")
            finally:
                # Unregister the temp view
                duckdb_conn.unregister(temp_view)
        elif nw_data.implementation == nw.Implementation.IBIS:
            # Already Ibis, use the standard Ibis API
            native_table = cast(ibis.Table, nw_data.to_native())

            if table_name not in self.conn.list_tables():
                if self.auto_create_tables:
                    self.conn.create_table(table_name, native_table)
                else:
                    raise ValueError(f"Table '{table_name}' does not exist and auto_create_tables is False")
            else:
                self.conn.insert(table_name, native_table)
        else:
            raise ValueError(
                f"Unsupported implementation: {nw_data.implementation}. "
                f"Expected POLARS or IBIS."
            )
