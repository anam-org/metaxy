"""Apache Iceberg storage implemented with PyIceberg."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.io_handler import IOHandler
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.table_ref import TableRef
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
from metaxy.models.types import FeatureKey
from metaxy.utils import collect_to_polars

TableIdentifier = NewType("TableIdentifier", tuple[str, str])
"""A ``(namespace, table_name)`` pair used by PyIceberg to locate a table within a catalog."""


def _strip_casts() -> Callable[[Any], Any]:
    """Unwrap ``CAST(x AS type)`` → ``x`` to match PyIceberg's
    `row filter syntax <https://py.iceberg.apache.org/row-filter-syntax/>`_
    which accepts bare literals but not SQL casts.
    """
    from sqlglot import exp

    def _transform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Cast):
            return node.this
        return node

    return _transform


@public
class IcebergMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for IcebergMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.polars.handlers.iceberg.IcebergMetadataStore"

        [stores.dev.config]
        warehouse = "/path/to/warehouse"
        namespace = "metaxy"

        [stores.dev.config.catalog_properties]
        type = "sql"
        ```
    """

    warehouse: str | Path = Field(
        description="Warehouse directory or URI where Iceberg tables are stored.",
    )
    namespace: str = Field(
        default="metaxy",
        description="Iceberg namespace for feature tables.",
    )
    catalog_name: str = Field(
        default="metaxy",
        description="Name of the Iceberg catalog.",
    )
    catalog_properties: dict[str, str] | None = Field(
        default=None,
        description="Properties passed to pyiceberg.catalog.load_catalog.",
    )
    auto_create_namespace: bool = Field(
        default=True,
        description="Automatically create the namespace on first write if it does not exist.",
    )


class IcebergStorageConfig(StorageConfig):
    """Storage configuration for Apache Iceberg tables in the warehouse at `location`.

    Attributes:
        namespace: Iceberg namespace for feature tables.
        catalog_name: Name of the PyIceberg catalog instance.
        catalog_properties: Properties passed to `pyiceberg.catalog.load_catalog`.
            Defaults to a SQLite catalog inside the warehouse.
        auto_create_namespace: Create the namespace on first write if missing.
    """

    namespace: str = "metaxy"
    catalog_name: str = "metaxy"
    catalog_properties: dict[str, str] | None = None
    auto_create_namespace: bool = True

    def resolve(self, key: FeatureKey) -> IcebergTableIdentifier:
        return IcebergTableIdentifier(storage=self, table_name=key.table_name)

    def load_catalog(self) -> Catalog:
        """Load the PyIceberg catalog of this storage."""
        from pyiceberg.catalog import load_catalog

        if is_local_path(self.location):
            Path(self.location).mkdir(parents=True, exist_ok=True)
        return load_catalog(
            self.catalog_name,
            **(
                self.catalog_properties
                or {
                    "type": "sql",
                    "uri": f"sqlite:///{self.location}/catalog.db",
                    "warehouse": f"file://{self.location}",
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class IcebergTableIdentifier(TableRef):
    """Identifies an Iceberg table by its storage and name."""

    storage: IcebergStorageConfig
    table_name: str

    @property
    def identifier(self) -> TableIdentifier:
        return TableIdentifier((self.storage.namespace, self.table_name))

    def table_parts(self) -> tuple[str, ...]:
        return self.identifier


class IcebergHandler(IOHandler[None, IcebergTableIdentifier]):
    """Reads and writes Apache Iceberg tables with PyIceberg and Polars.

    Catalogs are loaded on first use and closed when the engine closes.
    """

    def __init__(self) -> None:
        self._catalogs: dict[str, Catalog] = {}

    def can_handle(self, storage_config: StorageConfig) -> bool:
        return isinstance(storage_config, IcebergStorageConfig)

    def catalog(self, storage: IcebergStorageConfig) -> Catalog:
        """Return the catalog of `storage`, loading it on first use."""
        key = storage.model_dump_json()
        if key not in self._catalogs:
            self._catalogs[key] = storage.load_catalog()
        return self._catalogs[key]

    def on_close(self, conn: None) -> None:  # noqa: ARG002
        for catalog in self._catalogs.values():
            catalog.close()
        self._catalogs.clear()

    def read(
        self,
        conn: None,
        table_id: IcebergTableIdentifier,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        if not self.has_feature(conn, table_id):
            return None
        lazy = nw.from_native(pl.scan_iceberg(self.catalog(table_id.storage).load_table(table_id.identifier)))
        if filters:
            lazy = lazy.filter(*filters)
        if columns is not None:
            lazy = lazy.select(columns)
        return lazy

    def write(
        self,
        conn: None,  # noqa: ARG002
        table_id: IcebergTableIdentifier,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Append `df` to the table. `kwargs` are forwarded to `Table.append`."""
        from metaxy.utils._arrow_map import convert_structs_to_maps

        catalog = self.catalog(table_id.storage)
        if table_id.storage.auto_create_namespace:
            catalog.create_namespace_if_not_exists(table_id.storage.namespace)

        # Arrow `Utf8View` columns from Polars Enums are not supported
        frame = collect_to_polars(df).with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))
        arrow_table = convert_structs_to_maps(
            frame, columns=[METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD]
        ).to_arrow()
        _ensure_table(catalog, table_id.identifier, arrow_table.schema).append(arrow_table, **kwargs)

    def has_feature(
        self,
        conn: None,  # noqa: ARG002
        table_id: IcebergTableIdentifier,
    ) -> bool:
        return self.catalog(table_id.storage).table_exists(table_id.identifier)

    def drop(
        self,
        conn: None,
        table_id: IcebergTableIdentifier,
    ) -> None:
        if self.has_feature(conn, table_id):
            self.catalog(table_id.storage).drop_table(table_id.identifier)

    def delete(
        self,
        conn: None,
        table_id: IcebergTableIdentifier,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,  # noqa: ARG002
    ) -> None:
        """Delete rows with PyIceberg's `row filter syntax <https://py.iceberg.apache.org/row-filter-syntax/>`_."""
        lazy = self.read(conn, table_id)
        if lazy is None:
            return
        iceberg_table = self.catalog(table_id.storage).load_table(table_id.identifier)
        if not filters:
            iceberg_table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        predicate = narwhals_expr_to_sql_predicate(
            filters, lazy.collect_schema(), dialect="postgres", extra_transforms=_strip_casts()
        )
        iceberg_table.delete(delete_filter=predicate)

    def get_store_metadata(self, table_id: IcebergTableIdentifier) -> dict[str, Any]:
        return {"identifier": ".".join(table_id.identifier)}


def _ensure_table(catalog: Catalog, identifier: TableIdentifier, arrow_schema: Any) -> Table:
    """Create or evolve an Iceberg table to match the given Arrow schema."""
    table = catalog.create_table_if_not_exists(identifier, schema=arrow_schema)
    if table.schema().as_arrow() != arrow_schema:
        with table.update_schema() as update:
            update.union_by_name(arrow_schema)
    return table


@public
class IcebergMetadataStore(MetadataStore):
    """Apache Iceberg metadata store backed by [PyIceberg](https://py.iceberg.apache.org/).

    Stores feature metadata in Iceberg tables managed by a PyIceberg catalog.
    It uses the Polars versioning engine for provenance calculations.

    Example:

        ```py
        from metaxy.ext.polars.handlers.iceberg import IcebergMetadataStore

        store = IcebergMetadataStore(
            warehouse="s3://my-bucket/warehouse",
            namespace="ml_features",
            catalog_properties={"type": "glue"},
        )
        ```
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        warehouse: str | Path | None = None,  # noqa: ARG002
        *,
        namespace: str = "metaxy",  # noqa: ARG002
        catalog_name: str = "metaxy",  # noqa: ARG002
        catalog_properties: dict[str, str] | None = None,  # noqa: ARG002
        auto_create_namespace: bool = True,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    def __new__(
        cls,
        warehouse: str | Path | None = None,
        *,
        namespace: str = "metaxy",
        catalog_name: str = "metaxy",
        catalog_properties: dict[str, str] | None = None,
        auto_create_namespace: bool = True,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ) -> IcebergMetadataStore:
        if warehouse is None:
            raise ValueError("warehouse is required")

        from metaxy.ext.polars.engine import PolarsComputeEngine

        warehouse_str = str(warehouse)
        if is_local_path(warehouse_str):
            location = (
                Path(warehouse_str.removeprefix("file://").removeprefix("local://")).expanduser().resolve().as_posix()
            )
        else:
            location = warehouse_str.rstrip("/")

        instance = MetadataStore.__new__(cls)
        MetadataStore.__init__(
            instance,
            engine=PolarsComputeEngine(),
            storage=IcebergStorageConfig(
                location=location,
                namespace=namespace,
                catalog_name=catalog_name,
                catalog_properties=catalog_properties,
                auto_create_namespace=auto_create_namespace,
            ),
            fallback_stores=fallback_stores,
            versioning_engine="polars",
            **kwargs,
        )
        return instance

    @property
    def _iceberg_storage(self) -> IcebergStorageConfig:
        assert isinstance(self._storage, IcebergStorageConfig)
        return self._storage

    @property
    def namespace(self) -> str:
        return self._iceberg_storage.namespace

    @property
    def catalog(self) -> Catalog:
        """Return the PyIceberg catalog instance. Raises if the store is not open."""
        if not self._is_open:
            raise RuntimeError("IcebergMetadataStore is not open. Call open() first.")
        handler = self._engine.get_handler(self._iceberg_storage)
        assert isinstance(handler, IcebergHandler)
        return handler.catalog(self._iceberg_storage)

    def _table_identifier(self, feature_key: FeatureKey) -> TableIdentifier:
        return self._iceberg_storage.resolve(feature_key).identifier

    def display(self) -> str:
        """Return human-readable representation of the store."""
        return f"IcebergMetadataStore(warehouse={self._iceberg_storage.location})"

    @classmethod
    def config_model(cls) -> type[IcebergMetadataStoreConfig]:
        return IcebergMetadataStoreConfig
