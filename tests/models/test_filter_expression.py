from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy.models.filter_expression import (
    ColumnOperand,
    ComparisonExpression,
    FilterParseError,
    NarwhalsFilter,
    parse_filter_string,
)


@pytest.fixture()
def sample_lazy_frame() -> nw.LazyFrame[Any]:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [17, 25, 30, 45],
            "status": ["active", "active", "inactive", "deleted"],
            "is_active": [True, True, False, False],
            "deleted_at": [None, None, "2024-01-01", None],
        }
    )
    return nw.from_native(df.lazy())


def collect_ids(lf: nw.LazyFrame[Any], expr: nw.Expr) -> list[int]:
    result = lf.filter(expr).collect().to_native()
    return result.sort("id")["id"].to_list()


def test_parse_simple_comparison(sample_lazy_frame: nw.LazyFrame[Any]) -> None:
    expr = parse_filter_string("age > 25")
    ids = collect_ids(sample_lazy_frame, expr)
    assert ids == [3, 4]


def test_parse_compound_and(sample_lazy_frame: nw.LazyFrame[Any]) -> None:
    expr = parse_filter_string("age > 25 AND status = 'active'")
    ids = collect_ids(sample_lazy_frame, expr)
    assert ids == []


def test_parse_or_with_parentheses(sample_lazy_frame: nw.LazyFrame[Any]) -> None:
    expr = parse_filter_string("(age > 25 OR age < 20) AND status != 'deleted'")
    ids = collect_ids(sample_lazy_frame, expr)
    assert ids == [1, 3]


def test_parse_not(sample_lazy_frame: nw.LazyFrame[Any]) -> None:
    expr = parse_filter_string("NOT (status = 'deleted')")
    ids = collect_ids(sample_lazy_frame, expr)
    assert ids == [1, 2, 3]


def test_parse_boolean_and_null_literals(
    sample_lazy_frame: nw.LazyFrame[Any],
) -> None:
    expr = parse_filter_string("is_active = true AND deleted_at = NULL")
    ids = collect_ids(sample_lazy_frame, expr)
    assert ids == [1, 2]


def test_serialization_round_trip() -> None:
    filter_model = NarwhalsFilter.from_string("age <= 30 OR NOT is_active")
    dumped = filter_model.model_dump()
    assert dumped["expression"]["type"] == "logical"
    assert dumped["expression"]["operator"] == "or"
    assert dumped["source"] == "age <= 30 OR NOT is_active"

    restored = NarwhalsFilter.model_validate(dumped)
    assert isinstance(restored.to_expr(), nw.Expr)


def test_unsupported_expression_raises() -> None:
    with pytest.raises(FilterParseError, match="Unsupported expression"):
        parse_filter_string("age BETWEEN 20 AND 30")


def test_dotted_column_name_preserved() -> None:
    filter_model = NarwhalsFilter.from_string("metadata.owner = 'alice'")
    expression = filter_model.expression
    assert isinstance(expression, ComparisonExpression)
    assert isinstance(expression.left, ColumnOperand)
    assert expression.left.name == "metadata.owner"
