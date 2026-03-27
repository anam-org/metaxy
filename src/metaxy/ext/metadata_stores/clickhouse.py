"""ClickHouse metadata store — thin wrapper that composes ClickHouseEngine + IbisStorageConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import ibis.expr.datatypes as dt
import narwhals as nw
from narwhals.typing import Frame, FrameT
from pydantic import Field

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types
    from ibis.backends.sql import SQLBackend
    from ibis.expr.schema import Schema as IbisSchema

from metaxy._decorators import public
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.ibis import (
    IbisMetadataStoreConfig,
)
from metaxy.metadata_store.ibis_compute_engine import (
    IbisComputeEngine,
    IbisSQLHandler,
    IbisStorageConfig,
    IbisStoreBackcompat,
)
from metaxy.models.types import FeatureKey
from metaxy.versioning.ibis import IbisVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class ClickHouseVersioningEngine(IbisVersioningEngine):
    """Versioning engine for ClickHouse backend.

    Overrides concat_strings_over_groups to use ClickHouse-compatible
    syntax with collect() (groupArray) + arrayStringConcat.
    """

    def concat_strings_over_groups(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> FrameT:
        """Concatenate string values within groups using ClickHouse window functions.

        Uses collect() (groupArray) + arrayStringConcat instead of group_concat().over()
        which generates invalid SQL for ClickHouse.
        """
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table: ibis.expr.types.Table = cast(ibis.expr.types.Table, df.to_native())

        @ibis.udf.scalar.builtin
        def arrayStringConcat(arr: dt.Array[dt.String], sep: str) -> str:
            """ClickHouse arrayStringConcat() function."""
            ...

        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        window = ibis.window(
            group_by=group_by_columns,
            order_by=[ibis_table[col] for col in effective_order_by],
        )

        arr_expr = ibis_table[source_column].cast("string").collect().over(window)
        concat_expr = arrayStringConcat(arr_expr, separator)

        ibis_table = ibis_table.mutate(**{target_column: concat_expr})

        return cast(FrameT, nw.from_native(ibis_table))


# ---------------------------------------------------------------------------
# SQL handler: ClickHouse-specific Map <-> Struct transforms
# ---------------------------------------------------------------------------


class ClickHouseSQLHandler(IbisSQLHandler):
    """Handles ClickHouse-specific Map/Struct and JSON type transforms.

    On read, converts JSON columns to String and Map columns to Struct
    (metaxy system columns use field names from the feature spec; user columns
    discover keys from the data when ``auto_cast_struct_for_map`` is enabled).

    On write, converts Struct columns back to Map format when the target
    ClickHouse column is Map type.
    """

    def __init__(self, *, auto_create_tables: bool = False, auto_cast_struct_for_map: bool = True) -> None:
        super().__init__(auto_create_tables=auto_create_tables)
        self.auto_cast_struct_for_map = auto_cast_struct_for_map
        self._ch_schema_cache: dict[str, IbisSchema] = {}

    # --- transform hooks -----------------------------------------------------

    def transform_after_read(
        self,
        conn: SQLBackend,
        table: ibis.Table,
        table_name: str,
        key: FeatureKey,
    ) -> ibis.Table:
        """Transform ClickHouse-specific column types for PyArrow compatibility.

        Handles JSON to String cast. When ``enable_map_datatype`` is off, also
        converts metaxy Map(String, String) columns to named Struct.
        """
        import ibis.expr.datatypes as dt

        from metaxy.config import MetaxyConfig

        schema = table.schema()
        mutations: dict[str, Any] = {}

        for col_name, dtype in schema.items():
            if isinstance(dtype, dt.JSON):
                mutations[col_name] = table[col_name].cast("string")

            elif isinstance(dtype, dt.Map) and not MetaxyConfig.get().enable_map_datatype:
                from metaxy.models.constants import (
                    METAXY_DATA_VERSION_BY_FIELD,
                    METAXY_PROVENANCE_BY_FIELD,
                )

                if col_name in {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}:
                    mutations[col_name] = self._map_to_struct_expr(table, col_name, key)

        if not mutations:
            return table

        return table.mutate(**mutations)

    def transform_before_write(
        self,
        conn: SQLBackend,
        df: Frame,
        table_name: str,
        key: FeatureKey,  # noqa: ARG002
    ) -> Frame:
        """Transform Struct columns to Map format for ClickHouse Map columns.

        When ``enable_map_datatype`` is set, the base write path handles
        Struct→Map conversion via Arrow, so this is a no-op.
        """
        from metaxy.config import MetaxyConfig

        if MetaxyConfig.get().enable_map_datatype:
            return df

        if table_name not in conn.list_tables():
            return df

        ch_schema = self._get_cached_schema(conn, table_name)
        return self._transform_struct_to_map(df, ch_schema)

    def ibis_type_to_polars(self, ibis_type: Any) -> Any:
        """Convert Ibis data type to Polars, with ClickHouse Map support.

        Handles ``Map(K, V)`` as Polar's canonical map representation.
        """
        import ibis.expr.datatypes as dt
        import polars as pl

        if isinstance(ibis_type, dt.Map):
            key_pl = self.ibis_type_to_polars(ibis_type.key_type)
            value_pl = self.ibis_type_to_polars(ibis_type.value_type)
            return pl.List(pl.Struct({"key": key_pl, "value": value_pl}))

        return super().ibis_type_to_polars(ibis_type)

    # --- helpers --------------------------------------------------------------

    def _get_cached_schema(self, conn: SQLBackend, table_name: str) -> IbisSchema:
        """Get ClickHouse table schema, cached per handler instance."""
        if table_name not in self._ch_schema_cache:
            self._ch_schema_cache[table_name] = conn.table(table_name).schema()
        return self._ch_schema_cache[table_name]

    def _map_to_struct_expr(
        self,
        table: ibis.Table,
        col_name: str,
        key: FeatureKey,
    ) -> Any:
        """Convert a Map column to a named Struct using field names from the feature spec."""
        import ibis

        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()
        definition = graph.feature_definitions_by_key.get(key)
        if definition is None:
            return table[col_name].cast("string")

        field_names = [f.key.to_struct_key() for f in definition.spec.fields]
        if not field_names:
            return table[col_name].cast("string")

        map_col = table[col_name]
        struct_dict = {name: map_col.get(name, "") for name in field_names}
        return ibis.struct(struct_dict)

    def _transform_struct_to_map(self, df: Frame, ch_schema: IbisSchema) -> Frame:
        """Transform Struct columns to Map-compatible format for ClickHouse.

        When ``auto_cast_struct_for_map`` is True, transforms ALL DataFrame Struct
        columns whose target ClickHouse column is Map type. Otherwise only transforms
        metaxy system columns.
        """
        import ibis.expr.datatypes as dt

        from metaxy.models.constants import (
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE_BY_FIELD,
        )

        metaxy_struct_columns = {METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD}

        if self.auto_cast_struct_for_map:
            map_columns = {name for name, dtype in ch_schema.items() if isinstance(dtype, dt.Map)}
        else:
            map_columns = {
                name for name, dtype in ch_schema.items() if isinstance(dtype, dt.Map) and name in metaxy_struct_columns
            }

        if not map_columns:
            return df

        if df.implementation == nw.Implementation.IBIS:
            return self._transform_ibis_struct_to_map(df, map_columns, ch_schema)

        return self._transform_polars_struct_to_map(df, map_columns, ch_schema)

    def _transform_ibis_struct_to_map(
        self,
        df: Frame,
        map_columns: set[str],
        ch_schema: IbisSchema,
    ) -> Frame:
        """Transform Ibis Struct columns to Map format."""
        from typing import cast as typing_cast

        import ibis
        import ibis.expr.datatypes as dt

        ibis_table = typing_cast("ibis.Table", df.to_native())
        schema = ibis_table.schema()

        mutations: dict[str, ibis.Expr] = {}
        for col_name in map_columns:
            if col_name not in schema:
                continue

            col_dtype = schema[col_name]
            if not isinstance(col_dtype, dt.Struct):
                continue

            field_names = list(col_dtype.names)  # ty: ignore[invalid-argument-type]
            ch_map_dtype = ch_schema[col_name]
            target_value_type = ch_map_dtype.value_type  # ty: ignore[unresolved-attribute]

            if not field_names:
                mutations[col_name] = ibis.literal(
                    {},
                    type=dt.Map(dt.String(), target_value_type),  # ty: ignore[invalid-argument-type, missing-argument]
                )
                continue

            keys = ibis.array([ibis.literal(name) for name in field_names])
            values = ibis.array([ibis_table[col_name][name].cast(target_value_type) for name in field_names])
            mutations[col_name] = ibis.map(keys, values)

        if not mutations:
            return df

        result_table = ibis_table.mutate(**mutations)  # ty: ignore[invalid-argument-type]
        return nw.from_native(result_table, eager_only=False)

    def _transform_polars_struct_to_map(
        self,
        df: Frame,
        map_columns: set[str],
        ch_schema: IbisSchema,
    ) -> Frame:
        """Transform Polars Struct columns to List[Struct{key, value}] for ClickHouse Map."""
        import polars as pl

        from metaxy._utils import collect_to_polars

        pl_df = collect_to_polars(df)

        cols_to_transform: list[tuple[str, list[str], pl.DataType]] = []
        for col_name in map_columns:
            if col_name not in pl_df.columns:
                continue
            col_dtype = pl_df.schema[col_name]
            if not isinstance(col_dtype, pl.Struct):
                continue
            field_names = [f.name for f in col_dtype.fields]
            ch_map_dtype = ch_schema[col_name]
            target_pl_type = self.ibis_type_to_polars(
                ch_map_dtype.value_type  # ty: ignore[unresolved-attribute]
            )
            cols_to_transform.append((col_name, field_names, target_pl_type))

        if not cols_to_transform:
            return df

        transformations = []
        for col_name, field_names, target_type in cols_to_transform:
            if not field_names:
                empty_list_type = pl.List(pl.Struct({"key": pl.Utf8, "value": target_type}))
                transformations.append(pl.lit([], dtype=empty_list_type).alias(col_name))
                continue

            key_value_structs = [
                pl.when(pl.col(col_name).struct.field(field_name).is_not_null())
                .then(
                    pl.struct(
                        [
                            pl.lit(field_name).alias("key"),
                            pl.col(col_name).struct.field(field_name).cast(target_type).alias("value"),
                        ]
                    )
                )
                .otherwise(None)
                for field_name in field_names
            ]
            transformations.append(pl.concat_list(key_value_structs).list.drop_nulls().alias(col_name))

        pl_df = pl_df.with_columns(transformations)
        return nw.from_native(pl_df)


@public
class ClickHouseMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for ClickHouseMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.clickhouse.ClickHouseMetadataStore"

        [stores.dev.config]
        connection_string = "clickhouse://localhost:8443/default"
        hash_algorithm = "xxhash64"
        ```
    """

    auto_cast_struct_for_map: bool = Field(
        default=True,
        description="Auto-convert DataFrame Struct columns to Map format on write when the ClickHouse column is Map type. Metaxy system columns are always converted. Ignored when enable_map_datatype is set.",
    )


@public
class ClickHouseMetadataStore(IbisStoreBackcompat):
    """[ClickHouse](https://clickhouse.com/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Example: Connection String
        ```py
        store = ClickHouseMetadataStore("clickhouse://localhost:8443/default")
        ```

    Example: Connection Parameters
        ```py
        store = ClickHouseMetadataStore(
            connection_params={
                "host": "localhost",
                "port": 8443,
                "database": "default",
            },
            hash_algorithm=HashAlgorithm.XXHASH64,
        )
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,  # noqa: ARG002
        *,
        connection_params: dict[str, Any] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        auto_cast_struct_for_map: bool = True,  # noqa: ARG002
        table_prefix: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    def __new__(
        cls,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        auto_cast_struct_for_map: bool = True,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params")

        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        handler = ClickHouseSQLHandler(
            auto_create_tables=auto_create_tables,
            auto_cast_struct_for_map=auto_cast_struct_for_map,
        )

        engine = ClickHouseEngine(
            connection_string=connection_string,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handler=handler,
        )

        storage = [
            IbisStorageConfig(
                format="clickhouse",
                location=connection_string or "clickhouse",
                table_prefix=table_prefix or "",
            )
        ]

        instance = IbisStoreBackcompat.__new__(cls)
        MetadataStore.__init__(
            instance,
            engine=engine,
            storage=storage,
            fallback_stores=fallback_stores,
            auto_create_tables=auto_create_tables,
            **kwargs,
        )
        return instance

    @classmethod
    def from_config(cls, config: MetadataStoreConfig, **kwargs: Any) -> MetadataStore:  # type: ignore[override]
        from metaxy.config import MetaxyConfig
        from metaxy.metadata_store.fallback import FallbackStoreList

        config_dict = config.model_dump(exclude_unset=True, exclude={"fallback_stores"})
        store = cast(MetadataStore, cls(**config_dict, **kwargs))
        fallback_store_names = config.model_dump(exclude_unset=True).get("fallback_stores", [])
        if fallback_store_names:
            store.fallback_stores = FallbackStoreList(
                fallback_store_names,
                config=MetaxyConfig.get(),
                parent_hash_algorithm=store.hash_algorithm,
            )
        return store

    @classmethod
    def config_model(cls) -> type[ClickHouseMetadataStoreConfig]:
        return ClickHouseMetadataStoreConfig

    # -- backcompat properties (deprecated, will be removed in 0.2.0) --

    @property
    def _conn(self) -> Any:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_conn")
        return self._ibis_engine._conn


class ClickHouseEngine(IbisComputeEngine):
    """Compute engine for ClickHouse backends using Ibis."""

    versioning_engine_cls = ClickHouseVersioningEngine

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        auto_create_tables: bool = False,
        handler: ClickHouseSQLHandler | None = None,
    ) -> None:
        if connection_string is None and connection_params is None:
            raise ValueError(
                "Must provide either connection_string or connection_params. "
                "Example: connection_string='clickhouse://localhost:8443/default'"
            )

        super().__init__(
            connection_string=connection_string,
            backend="clickhouse" if connection_string is None else None,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handler=handler,
        )

    def _create_hash_functions(self) -> dict:
        import ibis

        hash_functions = {}

        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def lower(x: str) -> str:  # ty: ignore[empty-body]
            ...

        def md5_hash(col_expr):  # noqa: ANN001, ANN202
            return lower(HEX(MD5(col_expr.cast(str))))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        @ibis.udf.scalar.builtin
        def xxHash32(x: str) -> int:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def xxHash64(x: str) -> int:  # ty: ignore[empty-body]
            ...

        @ibis.udf.scalar.builtin
        def toString(x: int) -> str:  # ty: ignore[empty-body]
            ...

        def xxhash32_hash(col_expr):  # noqa: ANN001, ANN202
            return toString(xxHash32(col_expr))

        def xxhash64_hash(col_expr):  # noqa: ANN001, ANN202
            return toString(xxHash64(col_expr))

        hash_functions[HashAlgorithm.XXHASH32] = xxhash32_hash
        hash_functions[HashAlgorithm.XXHASH64] = xxhash64_hash

        return hash_functions

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    @property
    def sqlalchemy_url(self) -> str:
        from sqlalchemy.engine.url import make_url

        base_url = super().sqlalchemy_url
        url = make_url(base_url)

        is_secure = url.port == 8443 or (url.query and url.query.get("secure") == "True")

        if url.port == 8443:
            native_port = 9440
        elif url.port == 8123:
            native_port = 9000
        else:
            native_port = 9440 if is_secure else 9000

        url = url.set(
            drivername="clickhouse+native",
            port=native_port,
        )

        if is_secure:
            new_query = {k: v for k, v in (url.query or {}).items() if k != "protocol"}
            new_query["secure"] = "True"
            url = url.set(query=new_query)

        return url.render_as_string(hide_password=False)

    @classmethod
    def config_model(cls) -> type[ClickHouseMetadataStoreConfig]:
        return ClickHouseMetadataStoreConfig
