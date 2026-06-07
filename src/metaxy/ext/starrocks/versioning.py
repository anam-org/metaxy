"""StarRocks-native versioning support."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
from narwhals.typing import FrameT

from metaxy.ext.ibis.versioning import IbisHashFn, IbisVersioningEngine
from metaxy.utils.constants import TEMP_TABLE_NAME

if TYPE_CHECKING:
    import ibis
    from ibis import Expr as IbisExpr


@cache
def _json_object_fn(arg_count: int) -> Any:
    """Build a cached Ibis builtin UDF wrapper for JSON_OBJECT with a fixed arity."""
    import ibis
    import ibis.expr.datatypes as dt

    if arg_count < 0:
        raise ValueError("JSON object argument count cannot be negative")

    params = ", ".join(f"arg_{idx}: str" for idx in range(arg_count))
    namespace: dict[str, Any] = {"dt": dt}
    exec(f"def _json_object({params}) -> dt.JSON:\n    ...\n", namespace)  # noqa: S102
    return ibis.udf.scalar.builtin(name="json_object")(namespace["_json_object"])


@cache
def _json_string_fn() -> Any:
    """Return an Ibis builtin UDF wrapper for StarRocks JSON_STRING."""
    import ibis
    import ibis.expr.datatypes as dt

    namespace: dict[str, Any] = {"dt": dt}
    exec("def json_string(value: dt.JSON) -> str:\n    ...\n", namespace)  # noqa: S102
    return ibis.udf.scalar.builtin(name="json_string")(namespace["json_string"])


@cache
def _get_json_string_fn() -> Any:
    """Return an Ibis builtin UDF wrapper for StarRocks GET_JSON_STRING."""
    import ibis

    @ibis.udf.scalar.builtin(name="get_json_string")
    def get_json_string(json_str: str, json_path: str) -> str:  # ty: ignore[empty-body]
        ...

    return get_json_string


def _json_path_for_field(field_name: str) -> str:
    escaped = field_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'$."{escaped}"'


def build_json_string_expr(ibis_table: ibis.Table, field_columns: dict[str, str]) -> IbisExpr:
    """Build a StarRocks JSON string expression from source columns."""
    return build_json_string_from_exprs(
        {field_name: ibis_table[source_column] for field_name, source_column in field_columns.items()}
    )


def build_json_string_from_exprs(field_values: dict[str, Any]) -> IbisExpr:
    """Build a StarRocks JSON string expression from Ibis value expressions."""
    import ibis

    args: list[IbisExpr] = []
    for field_name, value_expr in field_values.items():
        args.append(ibis.literal(field_name))
        args.append(value_expr.cast("string"))
    return _json_string_fn()(_json_object_fn(len(args))(*args))


class StarRocksVersioningEngine(IbisVersioningEngine):
    """Versioning engine for StarRocks-backed Ibis frames."""

    @classmethod
    def implementation(cls) -> nw.Implementation:
        return nw.Implementation.IBIS

    def build_struct_column(
        self,
        df: FrameT,
        struct_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build Metaxy by-field metadata as a JSON string column."""
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table = cast(ibis.expr.types.Table, df.to_native())
        result = ibis_table.mutate(**{struct_name: build_json_string_expr(ibis_table, field_columns)})
        return cast(FrameT, nw.from_native(result, eager_only=False))

    @staticmethod
    def build_map_column(
        df: FrameT,
        col_name: str,
        field_columns: dict[str, str],
    ) -> FrameT:
        """Build Metaxy by-field metadata as a JSON string column."""
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table = cast(ibis.expr.types.Table, df.to_native())
        result = ibis_table.mutate(**{col_name: build_json_string_expr(ibis_table, field_columns)})
        return cast(FrameT, nw.from_native(result, eager_only=False))

    def _extract_metadata_fields(self, df: FrameT, col_name: str, field_mapping: dict[str, str]) -> FrameT:
        """Extract fields from StarRocks JSON string metadata columns."""
        import ibis.expr.datatypes as dt
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table = cast(ibis.expr.types.Table, df.to_native())
        dtype = ibis_table.schema().get(col_name)

        if isinstance(dtype, (dt.String, dt.JSON)):
            json_source = ibis_table[col_name]
            if isinstance(dtype, dt.JSON):
                json_source = _json_string_fn()(json_source)

            get_json_string = _get_json_string_fn()
            mutations = {
                output_column: get_json_string(json_source, _json_path_for_field(field_name)).cast("string")
                for field_name, output_column in field_mapping.items()
            }
            return cast(FrameT, nw.from_native(ibis_table.mutate(**mutations), eager_only=False))

        return super()._extract_metadata_fields(df, col_name, field_mapping)

    def concat_strings_over_groups(
        self,
        df: FrameT,
        source_column: str,
        target_column: str,
        group_by_columns: list[str],
        order_by_columns: list[str],
        separator: str = "|",
    ) -> FrameT:
        """Concatenate strings with StarRocks GROUP_CONCAT semantics."""
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table = cast(ibis.expr.types.Table, df.to_native())
        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        window = ibis.window(
            group_by=group_by_columns,
            order_by=[ibis_table[col] for col in effective_order_by],
        )
        concat_expr = ibis_table[source_column].cast("string").group_concat(sep=separator).over(window)
        return cast(FrameT, nw.from_native(ibis_table.mutate(**{target_column: concat_expr}), eager_only=False))

    @staticmethod
    def keep_latest_by_group(
        df: FrameT,
        group_columns: list[str],
        timestamp_columns: list[str],
    ) -> FrameT:
        """Keep latest rows with ROW_NUMBER instead of argmax aggregation."""
        import ibis
        import ibis.expr.types

        assert df.implementation == nw.Implementation.IBIS, "Only Ibis DataFrames are accepted"
        ibis_table = cast(ibis.expr.types.Table, df.to_native())
        ordering_expr = ibis.coalesce(*[ibis_table[col] for col in timestamp_columns])
        window = ibis.window(
            group_by=group_columns,
            order_by=[ordering_expr.desc()],
        )
        ranked = ibis_table.mutate(**{TEMP_TABLE_NAME: ibis.row_number().over(window)})
        result = ranked.filter(ranked[TEMP_TABLE_NAME] == 0).drop(TEMP_TABLE_NAME)
        return cast(FrameT, nw.from_native(result, eager_only=False))


def create_starrocks_hash_functions() -> dict[Any, IbisHashFn]:
    """Create StarRocks hash functions for Ibis expressions."""
    import ibis

    from metaxy.versioning.types import HashAlgorithm

    @ibis.udf.scalar.builtin(name="md5")
    def md5(value: str) -> str:  # ty: ignore[empty-body]
        ...

    @ibis.udf.scalar.builtin(name="sha2")
    def sha2(value: str, hash_length: int) -> str:  # ty: ignore[empty-body]
        ...

    @ibis.udf.scalar.builtin(name="xx_hash3_64")
    def xx_hash3_64(value: str) -> int:  # ty: ignore[empty-body]
        ...

    def md5_hash(expr: IbisExpr) -> IbisExpr:
        return md5(cast(Any, expr).cast("string"))

    def sha256_hash(expr: IbisExpr) -> IbisExpr:
        return sha2(cast(Any, expr).cast("string"), 256)

    def xxh3_64_hash(expr: IbisExpr) -> IbisExpr:
        return xx_hash3_64(cast(Any, expr).cast("string")).cast("string")

    return {
        HashAlgorithm.XXH3_64: xxh3_64_hash,
        HashAlgorithm.MD5: md5_hash,
        HashAlgorithm.SHA256: sha256_hash,
    }
