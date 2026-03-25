"""Apache Iceberg metadata store implemented with PyIceberg."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType, overload

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.schema import Schema
    from pyiceberg.table import Table

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from packaging.version import Version
from pydantic import Field

from metaxy._decorators import public
from metaxy._utils import collect_to_polars
from metaxy.config import MetaxyConfig
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm

TableIdentifier = NewType("TableIdentifier", tuple[str, str])
"""A ``(namespace, table_name)`` pair used by PyIceberg to locate a table within a catalog."""


def _map_columns_from_iceberg_schema(schema: Schema) -> list[str]:
    """Return column names that have a native Map type in the Iceberg schema."""
    from pyiceberg.types import MapType

    return [field.name for field in schema.fields if isinstance(field.field_type, MapType)]


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
        type = "metaxy.ext.metadata_stores.iceberg.IcebergMetadataStore"

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


@public
class IcebergMetadataStore(MetadataStore):
    """Apache Iceberg metadata store backed by [PyIceberg](https://py.iceberg.apache.org/).

    Stores feature metadata in Iceberg tables managed by a PyIceberg catalog.
    It uses the Polars versioning engine for provenance calculations.

    !!! tip
        If Polars 1.39 or greater is installed, lazy Polars frames are sinked via
        `LazyFrame.sink_iceberg`, avoiding unnecessary materialization.

    Example:

        ```py
        from metaxy.ext.metadata_stores.iceberg import IcebergMetadataStore

        store = IcebergMetadataStore(
            warehouse="s3://my-bucket/warehouse",
            namespace="ml_features",
            catalog_properties={"type": "glue"},
        )
        ```
    """

    _should_warn_auto_create_tables = False
    versioning_engine_cls = PolarsVersioningEngine

    def __init__(
        self,
        warehouse: str | Path,
        *,
        namespace: str = "metaxy",
        catalog_name: str = "metaxy",
        catalog_properties: dict[str, str] | None = None,
        auto_create_namespace: bool = True,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Apache Iceberg metadata store.

        Args:
            warehouse: Warehouse directory or URI for Iceberg data files.
            namespace: Iceberg namespace for tables (Glue Database, SQL schema, etc.).
            catalog_name: Local identifier for the PyIceberg catalog instance.
            catalog_properties: Properties for [`pyiceberg.catalog.load_catalog`][pyiceberg.catalog.load_catalog].
            auto_create_namespace: Create the namespace on first write if it does not exist.
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][metaxy.metadata_store.base.MetadataStore].
        """
        self.namespace = namespace
        self.catalog_name = catalog_name
        self.auto_create_namespace = auto_create_namespace
        self._catalog: Catalog | None = None

        warehouse_str = str(warehouse)
        self._is_remote = not is_local_path(warehouse_str)

        if self._is_remote:
            self._warehouse_uri = warehouse_str.rstrip("/")
        else:
            if warehouse_str.startswith("file://"):
                warehouse_str = warehouse_str[7:]
            elif warehouse_str.startswith("local://"):
                warehouse_str = warehouse_str[8:]
            self._warehouse_uri = Path(warehouse_str).expanduser().resolve().as_posix()

        self._catalog_properties = catalog_properties or {
            "type": "sql",
            "uri": f"sqlite:///{self._warehouse_uri}/catalog.db",
            "warehouse": "file://" + self._warehouse_uri,
        }

        from metaxy.metadata_store.compute_engine import PolarsComputeEngine
        from metaxy.metadata_store.storage_config import StorageConfig

        engine = PolarsComputeEngine()
        storage = [StorageConfig(format="iceberg", location=self._warehouse_uri)]

        super().__init__(
            engine=engine,
            storage=storage,
            fallback_stores=fallback_stores,
            versioning_engine="polars",
            **kwargs,
        )

    # ===== MetadataStore abstract methods =====

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Check if feature exists in Iceberg catalog."""
        feature_key = self._resolve_feature_key(feature)
        return self.catalog.table_exists(self._table_identifier(feature_key))

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH32 by default."""
        return HashAlgorithm.XXHASH32

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        """Create Polars versioning engine for Iceberg store."""
        with self._create_polars_versioning_engine(plan) as engine:
            yield engine

    def _open(self, mode: AccessMode) -> None:  # noqa: ARG002
        if not self._is_remote:
            Path(self._warehouse_uri).mkdir(parents=True, exist_ok=True)
        from pyiceberg.catalog import load_catalog

        self._catalog = load_catalog(self.catalog_name, **self._catalog_properties)

    def _close(self) -> None:
        if self._catalog is not None:
            self._catalog.close()
            self._catalog = None

    # ===== Internal helpers =====

    @property
    def catalog(self) -> Catalog:
        """Return the PyIceberg catalog instance. Raises if the store is not open."""
        if self._catalog is None:
            msg = "IcebergMetadataStore is not open. Call open() first."
            raise RuntimeError(msg)
        return self._catalog

    def _table_identifier(self, feature_key: FeatureKey) -> TableIdentifier:
        return TableIdentifier((self.namespace, feature_key.table_name))

    def _ensure_namespace(self) -> None:
        """Create the namespace if auto_create_namespace is enabled."""
        if self.auto_create_namespace:
            self.catalog.create_namespace_if_not_exists(self.namespace)

    @overload
    def _cast_enum_to_string(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def _cast_enum_to_string(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def _cast_enum_to_string(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Cast Enum columns to String to avoid Arrow Utf8View incompatibility."""
        return frame.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

    def _ensure_table(self, identifier: TableIdentifier, arrow_schema: Any) -> Table:
        """Create or evolve an Iceberg table to match the given Arrow schema."""

        table: Table = self.catalog.create_table_if_not_exists(identifier, schema=arrow_schema)
        if table.schema().as_arrow() != arrow_schema:
            with table.update_schema() as update:
                update.union_by_name(arrow_schema)
        return table

    # ===== Storage operations =====

    def _write_feature(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Append metadata to the Iceberg table for a feature.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata
            **kwargs: Forwarded to `sink_iceberg` or `Table.append`.

        !!! tip
            If Polars 1.39 or greater is installed, lazy Polars frames are sinked via
            `LazyFrame.sink_iceberg`, avoiding unnecessary materialization.
        """
        self._ensure_namespace()
        identifier = self._table_identifier(feature_key)

        can_sink = (
            df.implementation == nw.Implementation.POLARS
            and isinstance(df, nw.LazyFrame)
            and Version(pl.__version__) >= Version("1.39.0")
        )

        if MetaxyConfig.get().enable_map_datatype:
            self._write_with_map_columns(df, identifier, **kwargs)
        elif can_sink:
            lf_native = df.to_native()
            assert isinstance(lf_native, pl.LazyFrame)
            arrow_schema = pl.DataFrame(schema=self._cast_enum_to_string(lf_native).collect_schema()).to_arrow().schema
            iceberg_table = self._ensure_table(identifier, arrow_schema)
            # sink_iceberg requires columns in the same order as the Iceberg table schema
            schema_col_order = [f.name for f in iceberg_table.schema().as_arrow()]
            self._cast_enum_to_string(lf_native).select(schema_col_order).sink_iceberg(
                iceberg_table, mode="append", **kwargs
            )
        else:
            df_polars = self._cast_enum_to_string(collect_to_polars(df))
            arrow_table = df_polars.to_arrow()
            iceberg_table = self._ensure_table(identifier, arrow_table.schema)
            iceberg_table.append(arrow_table, **kwargs)

    def _read_feature(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata stored in Iceberg for a single feature using lazy evaluation.

        Args:
            feature: Feature to read metadata for
            filters: List of Narwhals filter expressions
            columns: Subset of columns to return
            **kwargs: Backend-specific parameters (currently unused)
        """
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        identifier = self._table_identifier(feature_key)
        if not self.catalog.table_exists(identifier):
            return None

        iceberg_table = self.catalog.load_table(identifier)
        lf = pl.scan_iceberg(iceberg_table)

        if MetaxyConfig.get().enable_map_datatype:
            lf = self._read_map_columns(lf, iceberg_table)

        nw_lazy = nw.from_native(lf)

        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def _write_with_map_columns(
        self,
        df: Frame,
        identifier: TableIdentifier,
        **kwargs: Any,
    ) -> None:
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
        """Convert native Iceberg Map columns back to Polars Map dtype on read."""
        map_columns = _map_columns_from_iceberg_schema(iceberg_table.schema())
        if map_columns:
            from metaxy.versioning._arrow_map import convert_maps_to_polars_map

            return convert_maps_to_polars_map(lf, columns=map_columns)
        return lf

    def _drop_feature(self, feature_key: FeatureKey) -> None:
        """Drop the Iceberg table for the specified feature from the catalog."""
        identifier = self._table_identifier(feature_key)
        if self.catalog.table_exists(identifier):
            self.catalog.drop_table(identifier)

    def _delete_feature(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        """Delete rows from an Iceberg table using PyIceberg's `row filter syntax
        <https://py.iceberg.apache.org/row-filter-syntax/>`_."""
        identifier = self._table_identifier(feature_key)
        if not self.catalog.table_exists(identifier):
            return

        iceberg_table = self.catalog.load_table(identifier)

        if not filters:
            iceberg_table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        schema = self.read_feature_schema_from_store(feature_key)
        predicate = narwhals_expr_to_sql_predicate(
            filters,
            schema,
            dialect="postgres",
            extra_transforms=_strip_casts(),
        )
        iceberg_table.delete(delete_filter=predicate)

    def display(self) -> str:
        """Return human-readable representation of the store."""
        return f"IcebergMetadataStore(warehouse={self._warehouse_uri})"

    def _get_store_metadata_impl(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        resolved = self._resolve_feature_key(feature_key)
        return {"identifier": ".".join(self._table_identifier(resolved))}

    @classmethod
    def config_model(cls) -> type[IcebergMetadataStoreConfig]:
        return IcebergMetadataStoreConfig
