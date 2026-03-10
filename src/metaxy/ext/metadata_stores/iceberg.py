"""Apache Iceberg metadata store implemented with PyIceberg."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._decorators import public
from metaxy._utils import collect_to_polars
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


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
    layout: Literal["flat", "nested"] = Field(
        default="nested",
        description="Table naming layout ('nested' or 'flat').",
    )


@public
class IcebergMetadataStore(MetadataStore):
    """Apache Iceberg metadata store backed by [PyIceberg](https://py.iceberg.apache.org/).

    Stores feature metadata in Iceberg tables managed by a PyIceberg catalog.
    Uses the Polars versioning engine for provenance calculations.

    Example:

        ```py
        from metaxy.ext.metadata_stores.iceberg import IcebergMetadataStore

        store = IcebergMetadataStore(
            warehouse="/path/to/warehouse",
            namespace="metaxy",
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
        fallback_stores: list[MetadataStore] | None = None,
        layout: Literal["flat", "nested"] = "nested",
        **kwargs: Any,
    ) -> None:
        if layout not in ("flat", "nested"):
            raise ValueError(f"Invalid layout: {layout}. Must be 'flat' or 'nested'.")

        self.namespace = namespace
        self.catalog_name = catalog_name
        self.layout = layout
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
            self._warehouse_uri = str(Path(warehouse_str).expanduser().resolve())

        self._catalog_properties = catalog_properties or {
            "type": "sql",
            "uri": f"sqlite:///{self._warehouse_uri}/catalog.db",
            "warehouse": self._warehouse_uri,
        }

        super().__init__(
            fallback_stores=fallback_stores,
            versioning_engine="polars",
            **kwargs,
        )

    def _open(self, mode: AccessMode) -> None:  # noqa: ARG002
        from pyiceberg.catalog import load_catalog

        if not self._is_remote:
            Path(self._warehouse_uri).mkdir(parents=True, exist_ok=True)

        self._catalog = load_catalog(
            self.catalog_name,
            **self._catalog_properties,
        )
        self._catalog.create_namespace_if_not_exists(self.namespace)

    def _close(self) -> None:
        self._catalog = None

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            raise RuntimeError("IcebergMetadataStore is not open. Call open() first.")
        return self._catalog

    def _table_identifier(self, feature_key: FeatureKey) -> tuple[str, ...]:
        if self.layout == "nested":
            return (self.namespace, *feature_key.parts)
        return (self.namespace, feature_key.table_name)

    def _ensure_namespace_hierarchy(self, feature_key: FeatureKey) -> None:
        """Ensure all namespace levels exist for nested layout."""
        if self.layout != "nested" or len(feature_key.parts) <= 1:
            return
        # For nested layout with multi-part keys, intermediate parts become namespaces.
        # e.g. ("metaxy", "test_stores", "upstream_a") needs namespace ("metaxy", "test_stores")
        for i in range(1, len(feature_key.parts)):
            ns = (self.namespace, *feature_key.parts[:i])
            self.catalog.create_namespace_if_not_exists(ns)

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        feature_key = self._resolve_feature_key(feature)
        return self.catalog.table_exists(self._table_identifier(feature_key))

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        with self._create_polars_versioning_engine(plan) as engine:
            yield engine

    @overload
    def _cast_enum_to_string(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def _cast_enum_to_string(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def _cast_enum_to_string(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Cast Enum columns to String to avoid Arrow Utf8View incompatibility."""
        return frame.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

    def _write_feature(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        df_polars = self._cast_enum_to_string(collect_to_polars(df))
        arrow_table = df_polars.to_arrow()
        identifier = self._table_identifier(feature_key)

        self._ensure_namespace_hierarchy(feature_key)
        iceberg_table = self.catalog.create_table_if_not_exists(identifier, schema=arrow_table.schema)
        if iceberg_table.schema().as_arrow() != arrow_table.schema:
            with iceberg_table.update_schema() as update:
                update.union_by_name(arrow_table.schema)
        iceberg_table.append(arrow_table)

    def _read_feature(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        identifier = self._table_identifier(feature_key)
        if not self.catalog.table_exists(identifier):
            return None

        nw_lazy = nw.from_native(self.catalog.load_table(identifier).to_polars())

        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def _drop_feature(self, feature_key: FeatureKey) -> None:
        identifier = self._table_identifier(feature_key)
        if self.catalog.table_exists(identifier):
            self.catalog.drop_table(identifier)

    def _delete_feature(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None = None,
        *,
        with_feature_history: bool,
    ) -> None:
        identifier = self._table_identifier(feature_key)
        if not self.catalog.table_exists(identifier):
            return

        iceberg_table = self.catalog.load_table(identifier)

        if not filters:
            iceberg_table.delete()
            return

        # Read all data, filter out matching rows, overwrite
        nw_df = nw.from_native(iceberg_table.to_polars().collect())
        kept = nw_df.filter(~nw.all_horizontal(*filters, ignore_nulls=False))
        iceberg_table.overwrite(kept.to_native().to_arrow())

    def display(self) -> str:
        return f"IcebergMetadataStore(warehouse={self._warehouse_uri})"

    def _get_store_metadata_impl(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        resolved = self._resolve_feature_key(feature_key)
        return {"identifier": ".".join(self._table_identifier(resolved))}

    @classmethod
    def config_model(cls) -> type[IcebergMetadataStoreConfig]:
        return IcebergMetadataStoreConfig
