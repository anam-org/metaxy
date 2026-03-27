from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import ibis.expr.datatypes as dt
import narwhals as nw
from narwhals.typing import FrameT

if TYPE_CHECKING:
    import ibis
    import ibis.expr.types

    from metaxy.ext.clickhouse.config import ClickHouseMetadataStoreConfig

from metaxy.metadata_store.ibis_compute_engine import IbisComputeEngine
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


class ClickHouseEngine(IbisComputeEngine):
    """Compute engine for ClickHouse backends using Ibis."""

    versioning_engine_cls = ClickHouseVersioningEngine

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        auto_create_tables: bool = False,
        handler: Any | None = None,
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
        from metaxy.ext.clickhouse.config import ClickHouseMetadataStoreConfig

        return ClickHouseMetadataStoreConfig
