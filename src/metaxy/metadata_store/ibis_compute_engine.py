"""Ibis-based I/O adapter for SQL databases.

Provides ``IbisTableAdapter`` (the compute engine) and ``IbisSQLHandler``
(the native SQL storage handler).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.compute_engine import ComputeEngine
from metaxy.metadata_store.exceptions import (
    FeatureNotFoundError,
    HashAlgorithmNotSupportedError,
    StoreNotOpenError,
    TableNotFoundError,
)
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.storage_handler import StorageHandler
from metaxy.models.types import FeatureKey
from metaxy.versioning.ibis import IbisVersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types
    from ibis.backends.sql import SQLBackend

    from metaxy.metadata_store.types import AccessMode
    from metaxy.models.plan import FeaturePlan


class IbisStorageConfig(StorageConfig):
    """Storage configuration for Ibis-backed SQL databases."""

    table_prefix: str = ""


class DuckLakeStorageConfig(IbisStorageConfig):
    """Storage configuration for DuckLake-backed tables.

    After the engine ATTACHes the DuckLake catalog, tables are accessed
    via normal SQL — so IbisSQLHandler handles I/O unchanged.
    """


# ---------------------------------------------------------------------------
# Storage handler: Ibis SQL tables
# ---------------------------------------------------------------------------


class IbisSQLHandler(StorageHandler["SQLBackend"]):
    """Reads and writes native SQL tables via an Ibis connection."""

    # Subclasses can override to suppress auto_create_tables warning
    _should_warn_auto_create_tables: bool = True

    def __init__(self, *, auto_create_tables: bool = False) -> None:
        self.auto_create_tables = auto_create_tables

    # --- capability ----------------------------------------------------------

    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, IbisStorageConfig)

    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, IbisStorageConfig)

    # --- helpers -------------------------------------------------------------

    def _resolve_table_name(self, storage_config: StorageConfig, key: FeatureKey) -> str:
        prefix = storage_config.table_prefix if isinstance(storage_config, IbisStorageConfig) else ""
        base_name = key.table_name
        return f"{prefix}{base_name}" if prefix else base_name

    def _get_filtered_ibis_lazy(
        self,
        conn: SQLBackend,
        table_name: str,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        import ibis.common.exceptions

        try:
            ibis_table = conn.table(table_name)
        except ibis.common.exceptions.TableNotFound:
            return None

        ibis_table = self.transform_after_read(conn, ibis_table, table_name, key)

        nw_frame = nw.from_native(ibis_table, eager_only=False)
        if not isinstance(nw_frame, nw.LazyFrame):
            raise TypeError(f"Expected narwhals LazyFrame from Ibis table, got {type(nw_frame)}")
        nw_lazy: nw.LazyFrame[Any] = nw_frame

        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        return nw_lazy

    # --- CRUD ----------------------------------------------------------------

    def read(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        table_name = self._resolve_table_name(storage_config, key)
        nw_lazy = self._get_filtered_ibis_lazy(conn, table_name, key, filters=filters)
        if nw_lazy is None:
            return None
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)
        return nw_lazy

    def write(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        table_name = self._resolve_table_name(storage_config, key)
        df = self.transform_before_write(conn, df, table_name, key)

        if df.implementation == nw.Implementation.IBIS:
            df_to_insert = df.to_native()
        else:
            from metaxy._utils import collect_to_polars

            df_to_insert = collect_to_polars(df)

        try:
            conn.insert(table_name, obj=df_to_insert)  # ty: ignore[invalid-argument-type]
        except Exception as e:
            import ibis.common.exceptions

            if not isinstance(e, ibis.common.exceptions.TableNotFound):
                raise
            if self.auto_create_tables:
                if self._should_warn_auto_create_tables:
                    import warnings

                    warnings.warn(
                        f"AUTO_CREATE_TABLES is enabled - automatically creating table '{table_name}'. "
                        "Do not use in production! "
                        "Use proper database migration tools like Alembic for production deployments.",
                        UserWarning,
                        stacklevel=4,
                    )
                conn.create_table(table_name, obj=df_to_insert)
            else:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist. "
                    f"Enable auto_create_tables=True to automatically create tables, "
                    f"or use proper database migration tools like Alembic to create the table first."
                ) from e

    def has_feature(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> bool:
        table_name = self._resolve_table_name(storage_config, key)
        return table_name in conn.list_tables()

    def drop(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> None:
        table_name = self._resolve_table_name(storage_config, key)
        if table_name in conn.list_tables():
            conn.drop_table(table_name)

    def delete(
        self,
        conn: SQLBackend,
        storage_config: StorageConfig,
        key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,  # noqa: ARG002
    ) -> None:
        from metaxy.metadata_store.utils import (
            _extract_where_expression,
            _strip_table_qualifiers,
        )

        table_name = self._resolve_table_name(storage_config, key)
        filter_list = list(filters or [])

        if not filter_list:
            if table_name not in conn.list_tables():
                raise TableNotFoundError(f"Table '{table_name}' does not exist.")
            conn.truncate_table(table_name)
            return

        nw_lazy = self._get_filtered_ibis_lazy(conn, table_name, key, filters=filter_list or None)
        if nw_lazy is None:
            raise FeatureNotFoundError(f"Table '{table_name}' not found in store")

        import ibis

        ibis_filtered = nw_lazy.to_native()
        if not isinstance(ibis_filtered, ibis.expr.types.Table):
            raise TypeError(f"Expected nw_lazy.to_native() to return an Ibis Table, got {type(ibis_filtered)!r}")
        select_sql = str(ibis_filtered.compile())

        dialect = conn.name
        predicate = _extract_where_expression(select_sql, dialect=dialect)
        if predicate is None:
            raise ValueError(f"Cannot extract WHERE clause for DELETE on {self.__class__.__name__}")

        predicate = predicate.transform(_strip_table_qualifiers())
        where_clause = predicate.sql(dialect=dialect) if dialect else predicate.sql()

        delete_stmt = f"DELETE FROM {table_name} WHERE {where_clause}"
        conn.raw_sql(delete_stmt)  # ty: ignore[unresolved-attribute]

    def get_store_metadata(
        self,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> dict[str, Any]:
        return {"table_name": self._resolve_table_name(storage_config, key)}

    # --- transform hooks (override in subclasses) ----------------------------

    def transform_after_read(
        self,
        conn: SQLBackend,
        table: ibis.Table,
        table_name: str,
        key: FeatureKey,  # noqa: ARG002
    ) -> ibis.Table:
        """Transform Ibis table after reading. Override in subclasses."""
        return table

    def transform_before_write(
        self,
        conn: SQLBackend,
        df: Frame,
        table_name: str,
        key: FeatureKey,  # noqa: ARG002
    ) -> Frame:
        """Transform DataFrame before writing. Override in subclasses."""
        return df

    def ibis_type_to_polars(self, ibis_type: Any) -> Any:
        """Convert an Ibis data type to the corresponding Polars data type."""
        import ibis.expr.datatypes as dt
        import polars as pl

        try:
            return ibis_type.to_polars()
        except NotImplementedError:
            if isinstance(ibis_type, dt.Float16):
                return pl.Float32
            elif isinstance(ibis_type, dt.UUID):
                return pl.String
            elif isinstance(ibis_type, dt.JSON):
                return pl.String
            elif isinstance(ibis_type, dt.MACADDR):
                return pl.String
            elif isinstance(ibis_type, dt.INET):
                return pl.String
            elif isinstance(ibis_type, dt.GeoSpatial):
                return pl.Binary
            else:
                raise


# ---------------------------------------------------------------------------
# Backcompat stub base for Ibis store factories
# ---------------------------------------------------------------------------


def _backcompat_warn_attr(attr: str) -> None:
    """Emit a warning for backcompat store attributes."""
    import warnings

    from metaxy.metadata_store.warnings import LegacyMetadataStoreAttributeWarning

    warnings.warn(
        f"Accessing '{attr}' on a store instance is deprecated and will be removed in Metaxy 0.2.0. "
        f"Use 'store._engine.{attr}' instead.",
        LegacyMetadataStoreAttributeWarning,
        stacklevel=3,
    )


def _backcompat_warn_init(cls_name: str) -> None:
    """Emit a warning when constructing a legacy store class directly."""
    import warnings

    from metaxy.metadata_store.warnings import LegacyMetadataStoreInstantiationWarning

    warnings.warn(
        f"{cls_name} is deprecated and will be removed in Metaxy 0.2.0. "
        f"Use MetadataStore(engine=..., storage=[...]) directly.",
        LegacyMetadataStoreInstantiationWarning,
        stacklevel=3,
    )


class IbisStoreBackcompat(MetadataStore):
    """Deprecated backcompat shim for Ibis-backed store factories.

    Provides delegating properties so existing code that accesses
    ``.conn``, ``.get_table_name()``, etc. directly on a store instance
    keeps working.  All accessors emit ``DeprecationWarning``.

    Will be removed in 0.2.0.
    """

    @property
    def _ibis_engine(self) -> IbisComputeEngine:
        assert isinstance(self._engine, IbisComputeEngine)
        return self._engine

    @property
    def conn(self) -> SQLBackend:
        _backcompat_warn_attr("conn")
        return self._ibis_engine.conn

    @property
    def connection_params(self) -> dict[str, Any]:
        _backcompat_warn_attr("connection_params")
        return self._ibis_engine.connection_params

    @property
    def _sql_dialect(self) -> str | None:
        return self._ibis_engine.conn.name

    @property
    def backend(self) -> str | None:
        _backcompat_warn_attr("backend")
        return self._ibis_engine.backend

    def ibis_type_to_polars(self, ibis_type: Any) -> Any:
        _backcompat_warn_attr("ibis_type_to_polars")
        handler = self._engine.handler
        assert isinstance(handler, IbisSQLHandler)
        return handler.ibis_type_to_polars(ibis_type)

    def _get_filtered_ibis_lazy(
        self,
        feature: Any,
        *,
        filters: Sequence[nw.Expr] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        _backcompat_warn_attr("_get_filtered_ibis_lazy")
        from metaxy.models.types import ValidatedFeatureKeyAdapter

        key = ValidatedFeatureKeyAdapter.validate_python(feature)
        handler = self._engine.handler
        assert isinstance(handler, IbisSQLHandler)
        table_name = handler._resolve_table_name(self._storage[0], key)
        return handler._get_filtered_ibis_lazy(self._ibis_engine.conn, table_name, key, filters=filters)


# ---------------------------------------------------------------------------
# Ibis compute engine
# ---------------------------------------------------------------------------


class IbisComputeEngine(ComputeEngine, ABC):
    """Abstract compute engine for SQL backends using Ibis.

    Concrete subclasses (DuckDB, ClickHouse, PostgreSQL, BigQuery) must
    implement ``_create_hash_functions`` and may provide custom handlers.
    """

    versioning_engine_cls = IbisVersioningEngine

    def __init__(
        self,
        *,
        connection_string: str | None = None,
        backend: str | None = None,
        connection_params: dict[str, Any] | None = None,
        auto_create_tables: bool = False,
        handler: StorageHandler[Any] | None = None,
    ) -> None:
        self.connection_string = connection_string
        self.backend = backend
        self.connection_params = connection_params or {}
        self._conn: SQLBackend | None = None
        self._handler: StorageHandler[Any] = handler or IbisSQLHandler(auto_create_tables=auto_create_tables)

    @property
    def handler(self) -> StorageHandler[Any]:
        return self._handler

    @property
    def connection(self) -> SQLBackend:
        if self._conn is None:
            raise StoreNotOpenError("Ibis connection is not open. Store must be used as a context manager.")
        return self._conn

    # Keep the old name as an alias for backwards compat within this module
    @property
    def conn(self) -> SQLBackend:
        return self.connection

    # --- lifecycle -----------------------------------------------------------

    def open(self, mode: AccessMode) -> None:  # noqa: ARG002
        import ibis

        if self.connection_string:
            self._conn = ibis.connect(self.connection_string)  # ty: ignore[invalid-assignment]
        else:
            assert self.backend is not None, "backend must be set if connection_string is None"
            backend_module = getattr(ibis, self.backend)
            self._conn = backend_module.connect(**self.connection_params)

        self._handler.on_connection_opened(self._conn)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.disconnect()
        self._conn = None

    # --- properties ----------------------------------------------------------

    @property
    def sqlalchemy_url(self) -> str:
        if self.connection_string:
            return self.connection_string
        raise ValueError(
            "SQLAlchemy URL not available. Store was initialized with backend + connection_params "
            "instead of a connection string."
        )

    # --- hashing / versioning ------------------------------------------------

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.MD5

    def validate_hash_algorithm_support(self, algorithm: HashAlgorithm) -> None:
        hash_functions = self._create_hash_functions()
        if algorithm not in hash_functions:
            supported = [algo.value for algo in hash_functions]
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{algorithm.value}' not supported. Supported algorithms: {', '.join(supported)}"
            )

    @abstractmethod
    def _create_hash_functions(self) -> dict:
        """Return a mapping of HashAlgorithm to Ibis hash function callables.

        Must be implemented by each concrete backend.
        """
        ...

    @contextmanager
    def create_versioning_engine(self, plan: FeaturePlan) -> Iterator[IbisVersioningEngine]:
        if self._conn is None:
            raise RuntimeError("Cannot create versioning engine: adapter is not open.")

        hash_functions = self._create_hash_functions()
        engine = self.versioning_engine_cls(
            plan=plan,
            hash_functions=hash_functions,  # ty: ignore[unknown-argument]
        )
        try:
            yield engine  # ty:ignore[invalid-yield]
        finally:
            pass

    # --- display / config ----------------------------------------------------

    def display(self) -> str:
        from metaxy.metadata_store.utils import sanitize_uri

        backend_info = self.connection_string or f"{self.backend}"
        sanitized_info = sanitize_uri(backend_info)
        return f"{self.__class__.__name__}(backend={sanitized_info})"

    @classmethod
    def config_model(cls) -> type[IbisMetadataStoreConfig]:
        return IbisMetadataStoreConfig
