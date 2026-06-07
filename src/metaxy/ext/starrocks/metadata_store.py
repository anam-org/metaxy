"""StarRocks metadata stores."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._decorators import experimental, public
from metaxy.ext.ibis.metadata_store import IbisMetadataStore, IbisMetadataStoreConfig
from metaxy.ext.polars.versioning import PolarsVersioningEngine
from metaxy.ext.starrocks.versioning import (
    StarRocksVersioningEngine,
    build_json_string_from_exprs,
    create_starrocks_hash_functions,
)
from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError, TableNotFoundError
from metaxy.metadata_store.types import AccessMode
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_DELETED_AT,
    METAXY_FEATURE_VERSION,
    METAXY_PROVENANCE_BY_FIELD,
    METAXY_UPDATED_AT,
)
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.utils import collect_to_polars
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStore

_METAXY_JSON_COLUMNS = frozenset({METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD})
_LATEST_TABLE_SUFFIX = "__metaxy_latest"
_LATEST_LIFECYCLE_COLUMN = "metaxy_latest_lifecycle_at"


def _install_pymysql_mysql_db_shim() -> None:
    """Let Ibis' MySQL backend import MySQLdb without native mysqlclient."""
    try:
        __import__("MySQLdb")
    except ImportError:
        try:
            import pymysql
        except ImportError as exc:
            raise ImportError("Install StarRocks dependencies with `pip install 'metaxy[starrocks]'`.") from exc

        pymysql.install_as_MySQLdb()


def _quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _format_identifier_list(identifiers: Sequence[str]) -> str:
    return ", ".join(_quote_identifier(identifier) for identifier in identifiers)


@public
@experimental
class StarRocksMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for StarRocks metadata stores."""

    replication_num: int | None = Field(
        default=1,
        description="StarRocks table replication_num property. Use 1 for single-node development clusters.",
    )
    buckets: int | None = Field(
        default=None,
        description="Optional StarRocks bucket count. If omitted, StarRocks chooses automatically.",
    )
    group_concat_max_len: int = Field(
        default=16_777_216,
        description="Session group_concat_max_len used for aggregation provenance.",
    )
    enable_latest_aggregate_table: bool = Field(
        default=True,
        description="Whether the optimized store maintains Aggregate-table latest-row projections.",
    )


class _StarRocksBaseMetadataStore(IbisMetadataStore):
    """Shared StarRocks storage behavior."""

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        replication_num: int | None = 1,
        buckets: int | None = None,
        group_concat_max_len: int = 16_777_216,
        enable_latest_aggregate_table: bool = False,
        **kwargs: Any,
    ):
        if connection_string is None and connection_params is None:
            raise ValueError(
                "Must provide either connection_string or connection_params. "
                "Example: connection_string='mysql://root@127.0.0.1:9030/metaxy'"
            )

        if connection_params is not None:
            connection_params = dict(connection_params)
            connection_params.setdefault("port", 9030)

        self.replication_num = replication_num
        self.buckets = buckets
        self.group_concat_max_len = group_concat_max_len
        self.enable_latest_aggregate_table = enable_latest_aggregate_table
        self._decode_json_reads = True

        super().__init__(
            connection_string=connection_string,
            backend="mysql",
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            **kwargs,
        )

    @classmethod
    def config_model(cls) -> type[StarRocksMetadataStoreConfig]:
        return StarRocksMetadataStoreConfig

    def _open(self, mode: AccessMode) -> None:
        _install_pymysql_mysql_db_shim()
        super()._open(mode)
        self.conn.raw_sql(f"SET SESSION group_concat_max_len = {self.group_concat_max_len}")  # ty: ignore[unresolved-attribute]

    @staticmethod
    def _is_table_not_found_error(e: Exception) -> bool:
        import ibis.common.exceptions

        if isinstance(e, ibis.common.exceptions.TableNotFound):
            return True
        message = str(e).lower()
        return "doesn't exist" in message or "unknown table" in message or "table not found" in message

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXH3_64

    def _properties_sql(self) -> str:
        properties: list[str] = []
        if self.replication_num is not None:
            properties.append(f'"replication_num" = "{self.replication_num}"')
        if not properties:
            return ""
        return f"\nPROPERTIES ({', '.join(properties)})"

    def _distribution_sql(self, key_columns: Sequence[str]) -> str:
        buckets_sql = f" BUCKETS {self.buckets}" if self.buckets is not None else ""
        return f"DISTRIBUTED BY HASH({_format_identifier_list(key_columns)}){buckets_sql}"

    def _column_sql_type(self, column_name: str, dtype: Any, *, is_key: bool) -> str:
        if column_name in _METAXY_JSON_COLUMNS:
            return "STRING"
        if dtype == nw.String:
            return "VARCHAR(255)" if is_key else "STRING"
        if dtype == nw.Boolean:
            return "BOOLEAN"
        if dtype in {nw.Int8, nw.Int16, nw.Int32, nw.UInt8, nw.UInt16, nw.UInt32}:
            return "INT"
        if dtype in {nw.Int64, nw.UInt64}:
            return "BIGINT"
        if dtype == nw.Float32:
            return "FLOAT"
        if dtype == nw.Float64:
            return "DOUBLE"
        if dtype == nw.Date:
            return "DATE"
        if isinstance(dtype, nw.Datetime):
            return "DATETIME"
        if isinstance(dtype, (nw.Struct, nw.List, nw.Array)):
            return "STRING"
        return "STRING"

    def _ordered_schema_items(self, schema: nw.Schema, key_columns: Sequence[str]) -> list[tuple[str, Any]]:
        key_set = set(key_columns)
        key_items = [(name, schema[name]) for name in key_columns if name in schema]
        value_items = [(name, dtype) for name, dtype in schema.items() if name not in key_set]
        return [*key_items, *value_items]

    def _resolve_table_key_columns(self, feature_key: FeatureKey, schema: nw.Schema) -> list[str]:
        if not self._is_system_table(feature_key):
            id_columns = [col for col in self._resolve_feature_plan(feature_key).feature.id_columns if col in schema]
            if id_columns:
                return id_columns

        names = schema.names()
        if not names:
            raise ValueError(f"Cannot create StarRocks table {feature_key.to_string()} from an empty schema")
        return [names[0]]

    def _build_create_duplicate_table_sql(
        self,
        table_name: str,
        schema: nw.Schema,
        key_columns: Sequence[str],
    ) -> str:
        column_lines = []
        key_set = set(key_columns)
        for column_name, dtype in self._ordered_schema_items(schema, key_columns):
            sql_type = self._column_sql_type(column_name, dtype, is_key=column_name in key_set)
            column_lines.append(f"  {_quote_identifier(column_name)} {sql_type}")

        columns_sql = ",\n".join(column_lines)
        return (
            f"CREATE TABLE IF NOT EXISTS {_quote_identifier(table_name)} (\n"
            f"{columns_sql}\n"
            ")\n"
            f"DUPLICATE KEY({_format_identifier_list(key_columns)})\n"
            f"{self._distribution_sql(key_columns)}"
            f"{self._properties_sql()}"
        )

    def _build_create_latest_table_sql(
        self,
        table_name: str,
        schema: nw.Schema,
        key_columns: Sequence[str],
    ) -> str:
        latest_table_name = self.get_latest_table_name(table_name)
        aggregate_key_columns = [*key_columns, METAXY_FEATURE_VERSION]
        column_lines = []
        for column_name in key_columns:
            sql_type = self._column_sql_type(column_name, schema[column_name], is_key=True)
            column_lines.append(f"  {_quote_identifier(column_name)} {sql_type}")
        column_lines.extend(
            [
                f"  {_quote_identifier(METAXY_FEATURE_VERSION)} VARCHAR(255)",
                f"  {_quote_identifier(_LATEST_LIFECYCLE_COLUMN)} DATETIME MAX",
            ]
        )
        columns_sql = ",\n".join(column_lines)
        return (
            f"CREATE TABLE IF NOT EXISTS {_quote_identifier(latest_table_name)} (\n"
            f"{columns_sql}\n"
            ")\n"
            f"AGGREGATE KEY({_format_identifier_list(aggregate_key_columns)})\n"
            f"{self._distribution_sql(aggregate_key_columns)}"
            f"{self._properties_sql()}"
        )

    def _warn_auto_create_table(self, table_name: str) -> None:
        if not self._should_warn_auto_create_tables:
            return
        warnings.warn(
            f"AUTO_CREATE_TABLES is enabled - automatically creating table '{table_name}'. "
            "Do not use in production! "
            "Use proper database migration tools like Alembic for production deployments.",
            UserWarning,
            stacklevel=4,
        )

    def _create_starrocks_table(self, feature_key: FeatureKey, table_name: str, df: Frame) -> list[str]:
        schema = df.collect_schema()
        key_columns = self._resolve_table_key_columns(feature_key, schema)
        statements = [self._build_create_duplicate_table_sql(table_name, schema, key_columns)]
        if self._should_maintain_latest_table(feature_key):
            statements.append(self._build_create_latest_table_sql(table_name, schema, key_columns))

        self._warn_auto_create_table(table_name)
        for statement in statements:
            self.conn.raw_sql(statement)  # ty: ignore[unresolved-attribute]
        return statements

    def _encode_json_columns(self, df: Frame) -> Frame:
        if df.implementation == nw.Implementation.IBIS:
            return self._encode_ibis_json_columns(df)
        return self._encode_polars_json_columns(df)

    def _encode_ibis_json_columns(self, df: Frame) -> Frame:
        import ibis.expr.datatypes as dt
        import ibis.expr.types

        ibis_table = cast(ibis.expr.types.Table, df.to_native())
        schema = ibis_table.schema()
        mutations: dict[str, Any] = {}

        for column_name in _METAXY_JSON_COLUMNS:
            dtype = schema.get(column_name)
            if isinstance(dtype, dt.Struct):
                field_values = {field_name: ibis_table[column_name][field_name] for field_name in dtype.names}
                mutations[column_name] = build_json_string_from_exprs(field_values)

        if not mutations:
            return df

        return nw.from_native(ibis_table.mutate(**mutations), eager_only=False)

    def _encode_polars_json_columns(self, df: Frame) -> Frame:
        pl_df = collect_to_polars(df)
        transforms = []

        for column_name in _METAXY_JSON_COLUMNS:
            if column_name not in pl_df.columns:
                continue
            dtype = pl_df.schema[column_name]
            if isinstance(dtype, pl.Struct):
                transforms.append(pl.col(column_name).struct.json_encode().alias(column_name))
            elif dtype == pl.Null:
                transforms.append(pl.col(column_name).cast(pl.String).alias(column_name))

        if transforms:
            pl_df = pl_df.with_columns(transforms)
        return nw.from_native(pl_df)

    def transform_before_write(self, df: Frame, feature_key: FeatureKey, table_name: str) -> Frame:
        """Encode Metaxy by-field columns to JSON strings before insertion."""
        _ = table_name
        if self._is_system_table(feature_key):
            return df
        return self._encode_json_columns(df)

    def _write_feature(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        table_name = self.get_table_name(feature_key)
        transformed = self.transform_before_write(df, feature_key, table_name)

        if table_name not in self.conn.list_tables():
            if not self.auto_create_tables:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist for feature {feature_key.to_string()}. "
                    "Enable auto_create_tables=True to automatically create tables, "
                    "or use proper database migration tools like Alembic to create the table first."
                )
            self._create_starrocks_table(feature_key, table_name, transformed)

        if transformed.implementation == nw.Implementation.IBIS:
            obj = transformed.to_native()
        else:
            obj = collect_to_polars(transformed)

        self.conn.insert(table_name, obj=obj)  # ty: ignore[invalid-argument-type]
        self._insert_latest_rows(feature_key, table_name, transformed)

    def _drop_feature(self, feature_key: FeatureKey) -> None:
        table_name = self.get_table_name(feature_key)
        latest_table_name = self.get_latest_table_name(table_name)

        if latest_table_name in self.conn.list_tables():
            self.conn.drop_table(latest_table_name)
        if table_name in self.conn.list_tables():
            self.conn.drop_table(table_name)

    def _delete_feature(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        super()._delete_feature(feature_key, filters, with_feature_history=with_feature_history)
        if self._should_maintain_latest_table(feature_key):
            self._refresh_latest_table(feature_key)

    def _should_maintain_latest_table(self, feature_key: FeatureKey) -> bool:
        return self.enable_latest_aggregate_table and not self._is_system_table(feature_key)

    @staticmethod
    def get_latest_table_name(table_name: str) -> str:
        return f"{table_name}{_LATEST_TABLE_SUFFIX}"

    def _insert_latest_rows(self, feature_key: FeatureKey, table_name: str, df: Frame) -> None:
        if not self._should_maintain_latest_table(feature_key):
            return

        schema = df.collect_schema()
        key_columns = self._resolve_table_key_columns(feature_key, schema)
        required_columns = {*key_columns, METAXY_FEATURE_VERSION, METAXY_UPDATED_AT, METAXY_DELETED_AT}
        if not required_columns.issubset(set(schema.names())):
            return

        latest_table_name = self.get_latest_table_name(table_name)
        if latest_table_name not in self.conn.list_tables():
            self.conn.raw_sql(self._build_create_latest_table_sql(table_name, schema, key_columns))  # ty: ignore[unresolved-attribute]

        latest_columns = [*key_columns, METAXY_FEATURE_VERSION, _LATEST_LIFECYCLE_COLUMN]
        if df.implementation == nw.Implementation.IBIS:
            import ibis
            import ibis.expr.types

            ibis_table = cast(ibis.expr.types.Table, df.to_native())
            latest_obj = ibis_table.mutate(
                **{
                    _LATEST_LIFECYCLE_COLUMN: ibis.coalesce(
                        ibis_table[METAXY_DELETED_AT], ibis_table[METAXY_UPDATED_AT]
                    )
                }
            ).select(latest_columns)
            self.conn.insert(latest_table_name, obj=latest_obj)
            return

        pl_df = collect_to_polars(df)
        latest_df = pl_df.with_columns(
            pl.coalesce(pl.col(METAXY_DELETED_AT), pl.col(METAXY_UPDATED_AT)).alias(_LATEST_LIFECYCLE_COLUMN)
        ).select(latest_columns)
        self.conn.insert(latest_table_name, obj=latest_df)  # ty: ignore[invalid-argument-type]

    def _refresh_latest_table(self, feature_key: FeatureKey) -> None:
        table_name = self.get_table_name(feature_key)
        latest_table_name = self.get_latest_table_name(table_name)
        if table_name not in self.conn.list_tables() or latest_table_name not in self.conn.list_tables():
            return

        schema = self._table_narwhals_schema(table_name)
        key_columns = self._resolve_table_key_columns(feature_key, schema)
        select_columns = _format_identifier_list([*key_columns, METAXY_FEATURE_VERSION])
        group_columns = _format_identifier_list([*key_columns, METAXY_FEATURE_VERSION])
        self.conn.raw_sql(f"TRUNCATE TABLE {_quote_identifier(latest_table_name)}")  # ty: ignore[unresolved-attribute]
        self.conn.raw_sql(  # ty: ignore[unresolved-attribute]
            f"INSERT INTO {_quote_identifier(latest_table_name)} "
            f"({select_columns}, {_quote_identifier(_LATEST_LIFECYCLE_COLUMN)}) "
            f"SELECT {select_columns}, MAX(COALESCE({_quote_identifier(METAXY_DELETED_AT)}, "
            f"{_quote_identifier(METAXY_UPDATED_AT)})) "
            f"FROM {_quote_identifier(table_name)} GROUP BY {group_columns}"
        )

    def _table_narwhals_schema(self, table_name: str) -> nw.Schema:
        ibis_schema = self.conn.table(table_name).schema()
        return nw.Schema({column_name: self.ibis_type_to_polars(dtype) for column_name, dtype in ibis_schema.items()})

    def _json_columns_to_parse(self, schema: nw.Schema) -> list[str]:
        return [column_name for column_name in _METAXY_JSON_COLUMNS if column_name in schema]

    def _parse_json_to_struct_columns(
        self,
        pl_df: pl.DataFrame,
        feature_key: FeatureKey,
        json_columns: Sequence[str],
    ) -> pl.DataFrame:
        _ = feature_key
        for column_name in json_columns:
            if column_name not in pl_df.columns or isinstance(pl_df.schema[column_name], pl.Struct):
                continue

            values = pl_df[column_name]
            non_null_values = values.drop_nulls()
            if non_null_values.len() == 0:
                continue

            normalized = non_null_values.cast(pl.Utf8).str.strip_chars()
            is_object_json = normalized.str.starts_with("{") & normalized.str.ends_with("}")
            is_json_null_literal = normalized == "null"
            if not (is_object_json | is_json_null_literal).all():
                continue

            try:
                decoded = values.str.json_decode(infer_schema_length=None)
            except pl.exceptions.PolarsError:
                continue

            if isinstance(decoded.dtype, pl.Struct):
                pl_df = pl_df.with_columns(decoded.alias(column_name))

        return pl_df

    def _cast_empty_system_struct_columns(
        self,
        pl_df: pl.DataFrame,
        feature_key: FeatureKey,
        json_columns: Sequence[str],
    ) -> pl.DataFrame:
        if pl_df.height != 0 or self._is_system_table(feature_key):
            return pl_df

        try:
            plan = self._resolve_feature_plan(feature_key)
        except (KeyError, RuntimeError):
            return pl_df

        field_names = [field_spec.key.to_struct_key() for field_spec in plan.feature.fields]
        expected_struct_dtype = pl.Struct({field_name: pl.String for field_name in field_names})
        casts = [
            pl.col(column_name).cast(expected_struct_dtype).alias(column_name)
            for column_name in _METAXY_JSON_COLUMNS
            if column_name in json_columns
            and column_name in pl_df.columns
            and not isinstance(pl_df.schema[column_name], pl.Struct)
        ]
        return pl_df.with_columns(casts) if casts else pl_df

    def _decode_json_frame(self, frame: nw.LazyFrame[Any], feature_key: FeatureKey) -> nw.LazyFrame[Any]:
        if self._is_system_table(feature_key):
            return frame

        schema = frame.collect_schema()
        json_columns = self._json_columns_to_parse(schema)
        if not json_columns:
            return frame

        pl_df = collect_to_polars(frame)
        pl_df = self._parse_json_to_struct_columns(pl_df, feature_key, json_columns)
        pl_df = self._cast_empty_system_struct_columns(pl_df, feature_key, json_columns)
        return nw.from_native(pl_df.lazy())

    @overload
    def read(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        allow_fallback: bool = True,
        with_feature_history: bool = False,
        with_sample_history: bool = False,
        include_soft_deleted: bool = False,
        with_store_info: Literal[False] = False,
    ) -> nw.LazyFrame[Any]: ...

    @overload
    def read(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        allow_fallback: bool = True,
        with_feature_history: bool = False,
        with_sample_history: bool = False,
        include_soft_deleted: bool = False,
        with_store_info: Literal[True],
    ) -> tuple[nw.LazyFrame[Any], MetadataStore]: ...

    def read(
        self,
        feature: CoercibleToFeatureKey,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        allow_fallback: bool = True,
        with_feature_history: bool = False,
        with_sample_history: bool = False,
        include_soft_deleted: bool = False,
        with_store_info: bool = False,
    ) -> nw.LazyFrame[Any] | tuple[nw.LazyFrame[Any], MetadataStore]:
        if with_store_info:
            result_with_store = super().read(
                feature,
                feature_version=feature_version,
                filters=filters,
                columns=columns,
                allow_fallback=allow_fallback,
                with_feature_history=with_feature_history,
                with_sample_history=with_sample_history,
                include_soft_deleted=include_soft_deleted,
                with_store_info=True,
            )
            if not self._decode_json_reads:
                return result_with_store

            feature_key = self._resolve_feature_key(feature)
            frame, store = result_with_store
            if store is not self:
                return result_with_store
            return self._decode_json_frame(frame, feature_key), store

        result = super().read(
            feature,
            feature_version=feature_version,
            filters=filters,
            columns=columns,
            allow_fallback=allow_fallback,
            with_feature_history=with_feature_history,
            with_sample_history=with_sample_history,
            include_soft_deleted=include_soft_deleted,
            with_store_info=False,
        )
        if not self._decode_json_reads:
            return result

        feature_key = self._resolve_feature_key(feature)
        return self._decode_json_frame(result, feature_key)

    @contextmanager
    def _raw_json_reads(self) -> Iterator[None]:
        previous = self._decode_json_reads
        self._decode_json_reads = False
        try:
            yield
        finally:
            self._decode_json_reads = previous

    def ibis_type_to_polars(self, ibis_type: Any) -> Any:
        import ibis.expr.datatypes as dt

        if isinstance(ibis_type, dt.Timestamp):
            return pl.Datetime("us")
        return super().ibis_type_to_polars(ibis_type)


@public
@experimental
class StarRocksMySQLMetadataStore(_StarRocksBaseMetadataStore):
    """Baseline StarRocks store over the MySQL wire protocol using Polars versioning."""

    versioning_engine_cls = PolarsVersioningEngine

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        enable_latest_aggregate_table: bool = False,
        **kwargs: Any,
    ):
        _ = enable_latest_aggregate_table
        super().__init__(
            connection_string=connection_string,
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            enable_latest_aggregate_table=False,
            **kwargs,
        )

    def native_implementation(self) -> nw.Implementation:
        return nw.Implementation.POLARS

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        yield self.versioning_engine_cls(plan=plan)  # ty: ignore[invalid-yield]

    def _create_hash_functions(self) -> dict[HashAlgorithm, Any]:
        return {}

    def _validate_hash_algorithm_support(self) -> None:
        supported = PolarsVersioningEngine.supported_hash_algorithms()
        if self.hash_algorithm not in supported:
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{self.hash_algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(a.value for a in sorted(supported, key=lambda a: a.value))}"
            )


@public
@experimental
class StarRocksMetadataStore(_StarRocksBaseMetadataStore):
    """Optimized StarRocks store with native versioning and latest-row acceleration."""

    versioning_engine_cls = StarRocksVersioningEngine

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        enable_latest_aggregate_table: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            connection_string=connection_string,
            connection_params=connection_params,
            fallback_stores=fallback_stores,
            enable_latest_aggregate_table=enable_latest_aggregate_table,
            **kwargs,
        )

    def _create_hash_functions(self) -> dict[HashAlgorithm, Any]:
        return create_starrocks_hash_functions()

    def _validate_hash_algorithm_support(self) -> None:
        supported = self._create_hash_functions()
        if self.hash_algorithm not in supported:
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{self.hash_algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(a.value for a in sorted(supported, key=lambda a: a.value))}"
            )

    def resolve_update(self, *args: Any, **kwargs: Any) -> Any:
        with self._raw_json_reads():
            return super().resolve_update(*args, **kwargs)

    def _keep_latest_by_group(
        self,
        *,
        df: nw.LazyFrame[Any],
        feature_key: FeatureKey,
        group_columns: list[str],
        timestamp_columns: list[str],
        feature_version: str | None,
    ) -> nw.LazyFrame[Any]:
        if (
            not self.enable_latest_aggregate_table
            or feature_version is None
            or self._is_system_table(feature_key)
            or df.implementation != nw.Implementation.IBIS
        ):
            return super()._keep_latest_by_group(
                df=df,
                feature_key=feature_key,
                group_columns=group_columns,
                timestamp_columns=timestamp_columns,
                feature_version=feature_version,
            )

        import ibis
        import ibis.expr.types

        table_name = self.get_table_name(feature_key)
        latest_table_name = self.get_latest_table_name(table_name)
        if latest_table_name not in self.conn.list_tables():
            return super()._keep_latest_by_group(
                df=df,
                feature_key=feature_key,
                group_columns=group_columns,
                timestamp_columns=timestamp_columns,
                feature_version=feature_version,
            )

        canonical = cast(ibis.expr.types.Table, df.to_native())
        latest = self.conn.table(latest_table_name)
        lifecycle_expr = ibis.coalesce(*[canonical[column_name] for column_name in timestamp_columns])
        predicates = [
            canonical[column_name] == latest[column_name] for column_name in [*group_columns, METAXY_FEATURE_VERSION]
        ]
        predicates.append(lifecycle_expr == latest[_LATEST_LIFECYCLE_COLUMN])
        joined = canonical.join(latest, predicates)
        result = joined.select([canonical[column_name] for column_name in canonical.columns])
        return self.versioning_engine_cls.keep_latest_by_group(
            df=nw.from_native(result, eager_only=False),
            group_columns=group_columns,
            timestamp_columns=timestamp_columns,
        )
