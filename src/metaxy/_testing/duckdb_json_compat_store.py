"""DuckDB-backed JSON-compatible store used in tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import ibis
import ibis.expr.datatypes as dt

from metaxy.metadata_store.ibis_json_compat import IbisJsonCompatStore
from metaxy.versioning.types import HashAlgorithm


class DuckDBJsonCompatStore(IbisJsonCompatStore):
    """IbisJsonCompatStore implementation for DuckDB used in tests."""

    def __init__(self, database: str, **kwargs: Any):
        super().__init__(
            backend="duckdb",
            connection_params={"database": database},
            hash_algorithm=HashAlgorithm.MD5,
            **kwargs,
        )

    def _create_hash_functions(self):
        import ibis

        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str: ...  # ty: ignore[invalid-return-type]

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str: ...  # ty: ignore[invalid-return-type]

        def md5_hash(col_expr):
            # DuckDB's MD5 returns a hex string; normalize to lowercase for consistency.
            return LOWER(MD5(col_expr.cast(str)))

        return {HashAlgorithm.MD5: md5_hash}

    def _get_json_unpack_exprs(
        self, json_column: str, field_names: list[str]
    ) -> dict[str, Any]:
        @ibis.udf.scalar.builtin
        def json_extract_string(_json: str, _path: str) -> str: ...  # ty: ignore[invalid-return-type]

        table = ibis._
        return {
            f"{json_column}__{field_name}": json_extract_string(
                table[json_column].cast("string"),
                ibis.literal(f"$.{field_name}"),
            )
            for field_name in field_names
        }

    def _get_json_pack_expr(
        self,
        struct_name: str,
        field_columns: Mapping[str, str],
    ) -> Any:
        table = ibis._

        @ibis.udf.scalar.builtin(output_type=dt.string)
        def to_json(_input) -> str: ...  # ty: ignore[invalid-return-type]

        keys_expr = ibis.array(
            [ibis.literal(name).cast("string") for name in sorted(field_columns)]
        )
        values_expr = ibis.array(
            [
                table[col_name].cast("string")
                for _, col_name in sorted(field_columns.items())
            ]
        )
        map_expr = ibis.map(keys_expr, values_expr)
        return to_json(map_expr)
