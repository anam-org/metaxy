"""Tests for SQL generation helpers."""

from datetime import datetime, timezone

import datafusion
import duckdb
import narwhals as nw
import polars as pl
import pyarrow as pa
import pytest
from metaxy_testing.predicate_cases import predicate_cases
from sqlglot import exp, parse_one
from syrupy.assertion import SnapshotAssertion

from metaxy.metadata_store.utils import (
    _strip_table_qualifiers,
    generate_sql,
    narwhals_expr_to_sql_predicate,
    unquote_identifiers,
)

PREDICATE_CASES = predicate_cases()


def duckdb_datafusion_table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array(list(range(10)), type=pa.int64()),
            "value": pa.array(list(range(10)), type=pa.int64()),
            "price": pa.array([1.0, 10.0, 3.0, 7.0, 2.5, 9.5, 4.0, 8.0, 0.5, 6.0]),
            "status": pa.array(
                [
                    "active",
                    "inactive",
                    "pending",
                    "active",
                    "inactive",
                    "pending",
                    "active",
                    "inactive",
                    "pending",
                    "active",
                ]
            ),
            "ts_tz": pa.array(
                [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    datetime(2024, 1, 3, tzinfo=timezone.utc),
                    datetime(2024, 1, 4, tzinfo=timezone.utc),
                    datetime(2024, 1, 5, tzinfo=timezone.utc),
                    datetime(2024, 1, 6, tzinfo=timezone.utc),
                    datetime(2024, 1, 7, tzinfo=timezone.utc),
                    datetime(2024, 1, 8, tzinfo=timezone.utc),
                    datetime(2024, 1, 9, tzinfo=timezone.utc),
                    datetime(2024, 1, 10, tzinfo=timezone.utc),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "ts_naive": pa.array(
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 4),
                    datetime(2024, 1, 5),
                    datetime(2024, 1, 6),
                    datetime(2024, 1, 7),
                    datetime(2024, 1, 8),
                    datetime(2024, 1, 9),
                    datetime(2024, 1, 10),
                ],
                type=pa.timestamp("us"),
            ),
        }
    )


def test_generate_sql_from_schema_dict() -> None:
    schema = nw.from_native(pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})).collect_schema()
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
    schema = nw.from_native(pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})).collect_schema()
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
    schema = nw.from_native(pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})).collect_schema()
    with pytest.raises(ValueError, match="at least one filter"):
        narwhals_expr_to_sql_predicate([], schema, dialect="duckdb")


def test_narwhals_expr_to_sql_predicate_strips_table_prefix() -> None:
    schema = nw.from_native(pl.DataFrame(schema={"date": pl.Date, "price": pl.Float64})).collect_schema()
    predicate = narwhals_expr_to_sql_predicate(nw.col("price") > 5, schema, dialect="duckdb")
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
        dialect="datafusion",
    )
    df = ctx.sql(f"SELECT * FROM prices WHERE {predicate}")
    batches = df.collect()
    assert sum(batch.num_rows for batch in batches) == 1


def test_narwhals_predicate_executes_in_datafusion_naive_timestamp() -> None:
    ctx = datafusion.SessionContext()
    ts_values = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
    ]
    batch = pa.record_batch(
        [pa.array([1.0, 10.0]), pa.array(ts_values, type=pa.timestamp("us"))],
        names=["price", "ts"],
    )
    ctx.register_record_batches("prices", [[batch]])

    schema = nw.from_native(pl.DataFrame(schema={"price": pl.Float64, "ts": pl.Datetime()})).collect_schema()
    predicate = narwhals_expr_to_sql_predicate(
        [nw.col("price") > 5, nw.col("ts") > datetime(2024, 1, 1)],
        schema,
        dialect="datafusion",
    )
    df = ctx.sql(f"SELECT * FROM prices WHERE {predicate}")
    batches = df.collect()
    assert sum(batch.num_rows for batch in batches) == 1


def test_narwhals_predicate_executes_in_datafusion_is_in() -> None:
    ctx = datafusion.SessionContext()
    batch = pa.record_batch(
        [pa.array(["active", "pending", "inactive"])],
        names=["status"],
    )
    ctx.register_record_batches("statuses", [[batch]])

    schema = nw.from_native(pl.DataFrame(schema={"status": pl.Utf8})).collect_schema()
    predicate = narwhals_expr_to_sql_predicate(
        [nw.col("status").is_in(["active", "pending"])],
        schema,
        dialect="datafusion",
    )
    df = ctx.sql(f"SELECT * FROM statuses WHERE {predicate}")
    batches = df.collect()
    assert sum(batch.num_rows for batch in batches) == 2


@pytest.mark.parametrize(
    "case",
    PREDICATE_CASES,
    ids=[case.name for case in PREDICATE_CASES],
)
@pytest.mark.parametrize("dialect", ["duckdb", "datafusion"])
def test_narwhals_expr_to_sql_predicate_outputs(
    case,
    dialect: str,
    snapshot: SnapshotAssertion,
) -> None:
    predicate = narwhals_expr_to_sql_predicate(
        case.exprs,
        case.schema,
        dialect=dialect,
    )
    assert predicate == snapshot


@pytest.mark.parametrize(
    "case",
    PREDICATE_CASES,
    ids=[case.name for case in PREDICATE_CASES],
)
def test_predicate_equality_duckdb_datafusion_snapshot(
    case,
    snapshot: SnapshotAssertion,
) -> None:
    table = duckdb_datafusion_table()
    schema = nw.from_native(table).collect_schema()

    duckdb_predicate = narwhals_expr_to_sql_predicate(
        case.exprs,
        schema,
        dialect="duckdb",
    )
    datafusion_predicate = narwhals_expr_to_sql_predicate(
        case.exprs,
        schema,
        dialect="datafusion",
    )

    con = duckdb.connect()
    con.register("t", table)
    duckdb_rows = con.execute(f"SELECT id FROM t WHERE {duckdb_predicate} ORDER BY id").fetchall()
    duckdb_ids = [row[0] for row in duckdb_rows]

    ctx = datafusion.SessionContext()
    ctx.register_record_batches("t", [table.to_batches()])
    df = ctx.sql(f"SELECT id FROM t WHERE {datafusion_predicate} ORDER BY id")
    batches = df.collect()
    datafusion_ids: list[int] = []
    for batch in batches:
        datafusion_ids.extend(batch.column(0).to_pylist())

    assert duckdb_ids == datafusion_ids
    assert {
        "case": case.name,
        "duckdb_predicate": duckdb_predicate,
        "datafusion_predicate": datafusion_predicate,
        "ids": duckdb_ids,
    } == snapshot


@pytest.mark.parametrize(
    "expr",
    [
        nw.col("value").cum_sum().over(order_by="value") > 5,
        nw.col("value").rank().over(order_by="value") > 1,
        nw.col("value").mean().over("group") > 3,
    ],
)
def test_narwhals_expr_to_sql_predicate_rejects_window_expressions(
    expr: nw.Expr,
) -> None:
    schema = nw.from_native(pl.DataFrame(schema={"value": pl.Int64, "group": pl.Utf8})).collect_schema()
    with pytest.raises(
        (RuntimeError, NotImplementedError),
        match="Could not extract WHERE clause|not supported",
    ):
        narwhals_expr_to_sql_predicate(expr, schema, dialect="duckdb")


# ===== Tests for unquote_identifiers transform =====


def test_unquote_identifiers_removes_quotes_from_columns() -> None:
    """Test that unquote_identifiers removes quotes from column identifiers."""
    sql = "\"status\" = 'active'"
    parsed = parse_one(sql, read="duckdb")
    transformed = parsed.transform(unquote_identifiers())
    result = transformed.sql(dialect="duckdb")

    # Should not have quotes around column name
    assert result == "status = 'active'"


def test_unquote_identifiers_handles_multiple_columns() -> None:
    """Test that unquote_identifiers works with multiple quoted columns."""
    sql = '"price" > 5 AND "status" = \'active\''
    parsed = parse_one(sql, read="duckdb")
    transformed = parsed.transform(unquote_identifiers())
    result = transformed.sql(dialect="duckdb")

    # Neither column should be quoted
    assert '"price"' not in result
    assert '"status"' not in result
    assert "price" in result
    assert "status" in result


def test_unquote_identifiers_preserves_unquoted_columns() -> None:
    """Test that already unquoted columns remain unchanged."""
    sql = "status = 'active'"
    parsed = parse_one(sql, read="duckdb")
    transformed = parsed.transform(unquote_identifiers())
    result = transformed.sql(dialect="duckdb")

    # Should remain unchanged
    assert result == "status = 'active'"


def test_unquote_identifiers_preserves_string_literals() -> None:
    """Test that string literals are not affected by unquote_identifiers."""
    sql = "\"status\" = 'active'"
    parsed = parse_one(sql, read="duckdb")
    transformed = parsed.transform(unquote_identifiers())
    result = transformed.sql(dialect="duckdb")

    # String literal should keep its quotes
    assert "'active'" in result


def test_unquote_identifiers_handles_complex_expressions() -> None:
    """Test unquote_identifiers with more complex SQL predicates."""
    sql = '("price" > 5 AND "status" IN (\'active\', \'pending\')) OR "value" IS NULL'
    parsed = parse_one(sql, read="duckdb")
    transformed = parsed.transform(unquote_identifiers())
    result = transformed.sql(dialect="duckdb")

    # All column names should be unquoted
    assert '"price"' not in result
    assert '"status"' not in result
    assert '"value"' not in result


# ===== Tests for extra_transforms parameter =====


def test_extra_transforms_single_transform() -> None:
    """Test passing a single transform to extra_transforms."""
    schema = nw.from_native(pl.DataFrame(schema={"status": pl.Utf8})).collect_schema()

    predicate = narwhals_expr_to_sql_predicate(
        nw.col("status") == "active",
        schema,
        dialect="datafusion",
        extra_transforms=unquote_identifiers(),
    )

    # Column name should be unquoted
    assert '"status"' not in predicate
    assert "status" in predicate


def test_extra_transforms_multiple_transforms() -> None:
    """Test passing multiple transforms as a sequence."""
    schema = nw.from_native(pl.DataFrame(schema={"status": pl.Utf8, "price": pl.Float64})).collect_schema()

    # Create a custom transform that uppercases column names (for testing)
    def uppercase_columns():
        def _uppercase(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Column) and isinstance(node.this, exp.Identifier):
                uppercased = node.copy()
                uppercased.this.set("this", node.this.this.upper())
                return uppercased
            return node

        return _uppercase

    predicate = narwhals_expr_to_sql_predicate(
        [nw.col("status") == "active", nw.col("price") > 5],
        schema,
        dialect="datafusion",
        extra_transforms=[unquote_identifiers(), uppercase_columns()],
    )

    # Columns should be unquoted and uppercased
    assert '"status"' not in predicate
    assert '"price"' not in predicate
    assert "STATUS" in predicate
    assert "PRICE" in predicate


def test_extra_transforms_none_uses_default_behavior() -> None:
    """Test that None extra_transforms behaves the same as before."""
    schema = nw.from_native(pl.DataFrame(schema={"status": pl.Utf8})).collect_schema()

    # Without extra_transforms
    predicate_default = narwhals_expr_to_sql_predicate(
        nw.col("status") == "active",
        schema,
        dialect="duckdb",
    )

    # With extra_transforms=None
    predicate_explicit_none = narwhals_expr_to_sql_predicate(
        nw.col("status") == "active",
        schema,
        dialect="duckdb",
        extra_transforms=None,
    )

    # Should be identical
    assert predicate_default == predicate_explicit_none


def test_extra_transforms_applied_after_strip_table_qualifiers() -> None:
    """Test that extra transforms are applied after table qualifiers are stripped."""
    schema = nw.from_native(pl.DataFrame(schema={"status": pl.Utf8})).collect_schema()

    predicate = narwhals_expr_to_sql_predicate(
        nw.col("status") == "active",
        schema,
        dialect="datafusion",
        extra_transforms=unquote_identifiers(),
    )

    # Should have no table qualifiers and no quotes
    assert "_metaxy_temp" not in predicate
    assert '"status"' not in predicate
    assert "status" in predicate


def test_extra_transforms_executes_in_datafusion() -> None:
    """Integration test: unquoted predicate executes correctly in DataFusion."""
    ctx = datafusion.SessionContext()
    batch = pa.record_batch(
        [pa.array(["active", "pending", "inactive"])],
        names=["status"],
    )
    ctx.register_record_batches("statuses", [[batch]])

    schema = nw.from_native(pl.DataFrame(schema={"status": pl.Utf8})).collect_schema()
    predicate = narwhals_expr_to_sql_predicate(
        nw.col("status") == "active",
        schema,
        dialect="datafusion",
        extra_transforms=unquote_identifiers(),
    )

    # Execute in DataFusion (which requires unquoted identifiers)
    df = ctx.sql(f"SELECT * FROM statuses WHERE {predicate}")
    batches = df.collect()
    assert sum(batch.num_rows for batch in batches) == 1
