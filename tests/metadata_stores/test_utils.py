"""Tests for shared metadata store utilities."""

from datetime import datetime, timezone

import datafusion
import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlglot import exp, parse_one
from syrupy.assertion import SnapshotAssertion

from metaxy.metadata_store.utils import (
    _strip_table_qualifiers,
    adapt_trino_query_for_datafusion,
    generate_sql,
    is_local_path,
    narwhals_expr_to_sql_predicate,
)


def test_is_local_path() -> None:
    """is_local_path should correctly detect local vs remote URIs."""
    assert is_local_path("./local/path")
    assert is_local_path("/absolute/path")
    assert is_local_path("relative/path")
    assert is_local_path("C:\\Windows\\Path")
    assert is_local_path("file:///absolute/path")
    assert is_local_path("local://path")

    assert not is_local_path("s3://bucket/path")
    assert not is_local_path("db://database-name")
    assert not is_local_path("http://remote-server/db")
    assert not is_local_path("https://remote-server/db")
    assert not is_local_path("gs://bucket/path")
    assert not is_local_path("az://container/path")


def test_generate_sql_from_schema_dict() -> None:
    schema = nw.from_native(
        pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})
    ).collect_schema()
    sql = generate_sql(
        lambda lf: lf.filter(nw.col("price") > 5),
        schema,
        dialect="duckdb",
    )

    parsed = parse_one(sql, read="duckdb")
    where_expr = parsed.args.get("where")
    assert where_expr is not None
    assert isinstance(where_expr.this, exp.GT)
    column = where_expr.this.find(exp.Column)
    assert column is not None
    assert column.name == "price"


def test_narwhals_expr_to_sql_predicate_combines_filters() -> None:
    schema = nw.from_native(
        pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})
    ).collect_schema()
    predicate = narwhals_expr_to_sql_predicate(
        [nw.col("price") > 5, nw.col("price") < 10],
        schema,
        dialect="duckdb",
    )
    parsed = parse_one(f"SELECT * FROM t WHERE {predicate}", read="duckdb")
    where_expr = parsed.args.get("where")
    assert where_expr is not None
    assert isinstance(where_expr.this, exp.And)


def test_narwhals_expr_to_sql_predicate_requires_filters() -> None:
    schema = nw.from_native(
        pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})
    ).collect_schema()
    with pytest.raises(ValueError, match="at least one filter"):
        narwhals_expr_to_sql_predicate([], schema, dialect="duckdb")


def test_narwhals_expr_to_sql_predicate_strips_table_prefix() -> None:
    schema = nw.from_native(
        pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})
    ).collect_schema()
    predicate = narwhals_expr_to_sql_predicate(
        nw.col("price") > 5, schema, dialect="duckdb"
    )
    assert "_metaxy_temp" not in predicate


def test_strip_table_qualifiers_handles_aliases() -> None:
    expr = parse_one('SELECT * FROM t WHERE t0."price" > 5', read="duckdb")
    where_expr = expr.args["where"].this
    stripped = where_expr.transform(_strip_table_qualifiers())
    assert "t0" not in stripped.sql(dialect="duckdb")


def test_narwhals_predicate_executes_in_datafusion() -> None:
    ctx = datafusion.SessionContext()
    ts_values = [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    ]
    batch = pa.record_batch(
        [pa.array([1.0, 10.0]), pa.array(ts_values, type=pa.timestamp("us", "UTC"))],
        names=["price", "ts"],
    )
    ctx.register_record_batches("prices", [[batch]])

    schema = nw.from_native(
        pl.DataFrame(schema={"price": pl.Float64, "ts": pl.Datetime(time_zone="UTC")})
    ).collect_schema()
    predicate = narwhals_expr_to_sql_predicate(
        [nw.col("price") > 5, nw.col("ts") > datetime(2024, 1, 1, tzinfo=timezone.utc)],
        schema,
        dialect="trino",
    )
    predicate = adapt_trino_query_for_datafusion(predicate)
    df = ctx.sql(f"SELECT * FROM prices WHERE {predicate}")
    batches = df.collect()
    assert sum(batch.num_rows for batch in batches) == 1


def test_adapt_trino_query_for_datafusion_rewrites_iso_timestamps() -> None:
    sql = (
        "\"ts\" = CAST(FROM_ISO8601_TIMESTAMP('2024-01-01T00:00:00+00:00') "
        "AS TIMESTAMP WITH TIME ZONE)"
    )
    adapted = adapt_trino_query_for_datafusion(sql)
    assert "FROM_ISO8601_TIMESTAMP" not in adapted
    assert "TIME ZONE" not in adapted
    assert adapted == "\"ts\" = CAST('2024-01-01T00:00:00' AS TIMESTAMP)"


def test_narwhals_expr_to_sql_predicate_outputs(
    snapshot: SnapshotAssertion,
) -> None:
    schema = nw.from_native(
        pl.DataFrame(
            schema={"price": pl.Float64, "status": pl.Utf8, "ts": pl.Datetime()}
        )
    ).collect_schema()
    filters = [
        nw.col("price") > 5,
        nw.col("status") == "active",
    ]
    duckdb_predicate = narwhals_expr_to_sql_predicate(
        filters,
        schema,
        dialect="duckdb",
    )
    trino_predicate = narwhals_expr_to_sql_predicate(
        filters,
        schema,
        dialect="trino",
    )
    assert {
        "duckdb": duckdb_predicate,
        "trino": trino_predicate,
    } == snapshot


@given(
    filters=st.lists(
        st.tuples(
            st.sampled_from(["<", "<=", ">", ">=", "==", "!="]),
            st.integers(min_value=-5, max_value=15),
        ),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=25, deadline=None)
def test_predicate_consistency_duckdb_datafusion(
    filters: list[tuple[str, int]],
) -> None:
    duckdb = pytest.importorskip("duckdb")
    table = pa.table(
        {
            "id": pa.array(list(range(10)), type=pa.int64()),
            "value": pa.array(list(range(10)), type=pa.int64()),
        }
    )
    schema = nw.from_native(table).collect_schema()

    exprs: list[nw.Expr] = []
    for op, val in filters:
        col = nw.col("value")
        if op == "<":
            exprs.append(col < val)
        elif op == "<=":
            exprs.append(col <= val)
        elif op == ">":
            exprs.append(col > val)
        elif op == ">=":
            exprs.append(col >= val)
        elif op == "==":
            exprs.append(col == val)
        else:
            exprs.append(col != val)

    duckdb_predicate = narwhals_expr_to_sql_predicate(
        exprs,
        schema,
        dialect="duckdb",
    )
    trino_predicate = narwhals_expr_to_sql_predicate(
        exprs,
        schema,
        dialect="trino",
    )
    datafusion_predicate = adapt_trino_query_for_datafusion(trino_predicate)

    con = duckdb.connect()
    con.register("t", table)
    duckdb_rows = con.execute(
        f"SELECT id FROM t WHERE {duckdb_predicate} ORDER BY id"
    ).fetchall()
    duckdb_ids = [row[0] for row in duckdb_rows]

    ctx = datafusion.SessionContext()
    ctx.register_record_batches("t", [table.to_batches()])
    df = ctx.sql(f"SELECT id FROM t WHERE {datafusion_predicate} ORDER BY id")
    batches = df.collect()
    datafusion_ids: list[int] = []
    for batch in batches:
        datafusion_ids.extend(batch.column(0).to_pylist())

    assert duckdb_ids == datafusion_ids
