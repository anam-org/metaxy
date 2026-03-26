"""Apache Iceberg metadata store implemented with PyIceberg."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from packaging.version import Version

from metaxy._decorators import public
from metaxy._utils import collect_to_polars
from metaxy.ext.iceberg.config import IcebergMetadataStoreConfig, TableIdentifier
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


def _strip_casts() -> Callable[[Any], Any]:
    """Unwrap ``CAST(x AS type)`` → ``x`` to match PyIceberg's row filter syntax."""
    from sqlglot import exp

    def _transform(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Cast):
            return node.this
        return node

    return _transform


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

    namespace: str
    catalog_name: str
    auto_create_namespace: bool
    _catalog: Catalog | None
    _is_remote: bool
    _warehouse_uri: str
    _catalog_properties: dict[str, str]

    def __init__(
        self,
        warehouse: str | Path | None = None,
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

        from metaxy.metadata_store.compute_engine import PolarsComputeEngine

        instance = MetadataStore.__new__(cls)
        instance.namespace = namespace
        instance.catalog_name = catalog_name
        instance.auto_create_namespace = auto_create_namespace
        instance._catalog = None

        warehouse_str = str(warehouse)
        instance._is_remote = not is_local_path(warehouse_str)

        if instance._is_remote:
            instance._warehouse_uri = warehouse_str.rstrip("/")
        else:
            if warehouse_str.startswith("file://"):
                warehouse_str = warehouse_str[7:]
            elif warehouse_str.startswith("local://"):
                warehouse_str = warehouse_str[8:]
            instance._warehouse_uri = Path(warehouse_str).expanduser().resolve().as_posix()

        instance._catalog_properties = catalog_properties or {
            "type": "sql",
            "uri": f"sqlite:///{instance._warehouse_uri}/catalog.db",
            "warehouse": "file://" + instance._warehouse_uri,
        }

        from metaxy.metadata_store.storage_config import StorageConfig

        MetadataStore.__init__(
            instance,
            engine=PolarsComputeEngine(),
            storage=[StorageConfig(format="iceberg", location=instance._warehouse_uri)],
            fallback_stores=fallback_stores,
            versioning_engine="polars",
            **kwargs,
        )
        return instance

    # ===== MetadataStore abstract methods =====

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        feature_key = self._resolve_feature_key(feature)
        return self.catalog.table_exists(self._table_identifier(feature_key))

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
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
        if self._catalog is None:
            msg = "IcebergMetadataStore is not open. Call open() first."
            raise RuntimeError(msg)
        return self._catalog

    def _table_identifier(self, feature_key: FeatureKey) -> TableIdentifier:
        return TableIdentifier((self.namespace, feature_key.table_name))

    def _ensure_namespace(self) -> None:
        if self.auto_create_namespace:
            self.catalog.create_namespace_if_not_exists(self.namespace)

    @overload
    def _cast_enum_to_string(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def _cast_enum_to_string(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def _cast_enum_to_string(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        return frame.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

    def _ensure_table(self, identifier: TableIdentifier, arrow_schema: Any) -> Table:
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
        self._ensure_namespace()
        identifier = self._table_identifier(feature_key)

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
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        identifier = self._table_identifier(feature_key)
        if not self.catalog.table_exists(identifier):
            return None

        nw_lazy = nw.from_native(pl.scan_iceberg(self.catalog.load_table(identifier)))

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
        filters: Sequence[nw.Expr] | None,
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

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        schema = self.read_feature_schema_from_store(feature_key)
        iceberg_table.delete(
            delete_filter=narwhals_expr_to_sql_predicate(
                filters,
                schema,
                dialect="postgres",
                extra_transforms=_strip_casts(),
            )
        )

    def display(self) -> str:
        return f"IcebergMetadataStore(warehouse={self._warehouse_uri})"

    def _get_store_metadata_impl(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        resolved = self._resolve_feature_key(feature_key)
        return {"identifier": ".".join(self._table_identifier(resolved))}

    @classmethod
    def config_model(cls) -> type[IcebergMetadataStoreConfig]:
        return IcebergMetadataStoreConfig
