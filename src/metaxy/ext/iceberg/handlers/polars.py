"""Iceberg storage handler using PyIceberg and Polars."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType, overload

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from packaging.version import Version

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.polars.storage_config import PolarsStorageConfig
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.storage_handler import StorageHandler
from metaxy.models.types import FeatureKey

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

TableIdentifier = NewType("TableIdentifier", tuple[str, str])


def _strip_casts() -> Callable[[Any], Any]:
    from sqlglot import exp

    def _transform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Cast):
            return node.this
        return node

    return _transform


class IcebergHandler(StorageHandler[None]):
    """Reads and writes Iceberg tables via PyIceberg and Polars.

    Manages its own PyIceberg catalog connection through lifecycle hooks.
    """

    def __init__(
        self,
        warehouse_uri: str,
        *,
        namespace: str,
        catalog_name: str,
        catalog_properties: dict[str, str],
        auto_create_namespace: bool,
        is_remote: bool,
    ) -> None:
        self._warehouse_uri = warehouse_uri
        self._namespace = namespace
        self._catalog_name = catalog_name
        self._catalog_properties = catalog_properties
        self._auto_create_namespace = auto_create_namespace
        self._is_remote = is_remote
        self._catalog: Catalog | None = None

    # -- Lifecycle hooks -----------------------------------------------------------

    def on_connection_opened(self, conn: None) -> None:
        if not self._is_remote:
            Path(self._warehouse_uri).mkdir(parents=True, exist_ok=True)
        from pyiceberg.catalog import load_catalog

        self._catalog = load_catalog(self._catalog_name, **self._catalog_properties)

    def on_connection_closing(self) -> None:
        if self._catalog is not None:
            self._catalog.close()
            self._catalog = None

    # -- Catalog access ------------------------------------------------------------

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            raise RuntimeError("IcebergHandler catalog is not open.")
        return self._catalog

    # -- StorageHandler interface ---------------------------------------------------

    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        return isinstance(storage_config, PolarsStorageConfig)

    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        return isinstance(storage_config, PolarsStorageConfig)

    def read(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        identifier = self._table_identifier(key)
        if not self.catalog.table_exists(identifier):
            return None

        iceberg_table = self.catalog.load_table(identifier)
        lf = pl.scan_iceberg(iceberg_table)

        from metaxy.config import MetaxyConfig

        if MetaxyConfig.get().enable_map_datatype:
            lf = self._read_map_columns(lf, iceberg_table)

        nw_lazy = nw.from_native(lf)
        if filters:
            nw_lazy = nw_lazy.filter(*filters)
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)
        return nw_lazy

    def write(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        self._ensure_namespace()
        identifier = self._table_identifier(key)

        from metaxy.config import MetaxyConfig

        if MetaxyConfig.get().enable_map_datatype:
            self._write_with_map_columns(df, identifier, **kwargs)
            return

        can_sink = (
            df.implementation == nw.Implementation.POLARS
            and isinstance(df, nw.LazyFrame)
            and Version(pl.__version__) >= Version("1.39.0")
        )

        if can_sink:
            lf_native = df.to_native()
            assert isinstance(lf_native, pl.LazyFrame)
            arrow_schema = pl.DataFrame(schema=self._cast_enum_to_string(lf_native).collect_schema()).to_arrow().schema
            iceberg_table = self._ensure_table(identifier, arrow_schema)
            schema_col_order = [f.name for f in iceberg_table.schema().as_arrow()]
            self._cast_enum_to_string(lf_native).select(schema_col_order).sink_iceberg(
                iceberg_table, mode="append", **kwargs
            )
        else:
            arrow_table = self._cast_enum_to_string(collect_to_polars(df)).to_arrow()
            iceberg_table = self._ensure_table(identifier, arrow_table.schema)
            iceberg_table.append(arrow_table, **kwargs)

    def has_feature(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> bool:
        return self.catalog.table_exists(self._table_identifier(key))

    def drop(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> None:
        identifier = self._table_identifier(key)
        if self.catalog.table_exists(identifier):
            self.catalog.drop_table(identifier)

    def delete(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        identifier = self._table_identifier(key)
        if not self.catalog.table_exists(identifier):
            return

        iceberg_table = self.catalog.load_table(identifier)
        if not filters:
            iceberg_table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        lf = self.read(None, storage_config, key)
        if lf is None:
            return
        iceberg_table.delete(
            delete_filter=narwhals_expr_to_sql_predicate(
                filters,
                lf.collect_schema(),
                dialect="postgres",
                extra_transforms=_strip_casts(),
            )
        )

    def get_store_metadata(
        self,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> dict[str, Any]:
        return {"identifier": ".".join(self._table_identifier(key))}

    # -- Internal helpers ----------------------------------------------------------

    def _table_identifier(self, key: FeatureKey) -> TableIdentifier:
        return TableIdentifier((self._namespace, key.table_name))

    def _ensure_namespace(self) -> None:
        if self._auto_create_namespace:
            self.catalog.create_namespace_if_not_exists(self._namespace)

    def _ensure_table(self, identifier: TableIdentifier, arrow_schema: Any) -> Table:
        table = self.catalog.create_table_if_not_exists(identifier, schema=arrow_schema)
        if table.schema().as_arrow() != arrow_schema:
            with table.update_schema() as update:
                update.union_by_name(arrow_schema)
        return table

    @overload
    def _cast_enum_to_string(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def _cast_enum_to_string(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def _cast_enum_to_string(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        return frame.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

    def _write_with_map_columns(self, df: Frame, identifier: TableIdentifier, **kwargs: Any) -> None:
        """Collect to Arrow and convert Struct *_by_field columns to native MapArray before writing."""
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
        from metaxy.versioning._arrow_map import convert_extension_maps_to_native, convert_structs_to_maps

        df_polars = self._cast_enum_to_string(collect_to_polars(df))
        df_polars = convert_structs_to_maps(
            df_polars, columns=[METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD]
        )
        arrow_table = convert_extension_maps_to_native(df_polars.to_arrow())
        iceberg_table = self._ensure_table(identifier, arrow_table.schema)
        iceberg_table.append(arrow_table, **kwargs)

    def _read_map_columns(self, lf: pl.LazyFrame, iceberg_table: Table) -> pl.LazyFrame:
        """Convert native Iceberg Map columns back to polars_map.Map on read."""
        from pyiceberg.types import MapType

        map_columns = [field.name for field in iceberg_table.schema().fields if isinstance(field.field_type, MapType)]
        if map_columns:
            from metaxy.versioning._arrow_map import convert_maps_to_polars_map

            return convert_maps_to_polars_map(lf, columns=map_columns)
        return lf
