from typing import Any

import narwhals as nw
import polars as pl
import pytest
import sqlglot

from metaxy.models.feature_spec import FeatureDep
from metaxy.models.filter_expression import (
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
    filter_model = NarwhalsFilter.model_validate("age <= 30 OR NOT is_active")
    dumped = filter_model.model_dump()
    assert dumped["expression"] == "age <= 30 OR NOT is_active"
    assert dumped["source"] == "age <= 30 OR NOT is_active"

    restored = NarwhalsFilter.model_validate(dumped["source"])
    assert isinstance(restored.to_expr(), nw.Expr)


def test_unsupported_expression_raises() -> None:
    with pytest.raises(FilterParseError, match="Unsupported expression"):
        parse_filter_string("age BETWEEN 20 AND 30")


def test_dotted_column_name_preserved() -> None:
    filter_model = NarwhalsFilter.model_validate("metadata.owner = 'alice'")
    expression = filter_model.expression
    assert isinstance(expression, sqlglot.exp.EQ)
    assert isinstance(expression.this, sqlglot.exp.Column)
    assert expression.this.sql() == "metadata.owner"


# Direct expression comparison tests - asserting parsed expression structure
def test_parsed_expression_equals_narwhals_expression_simple() -> None:
    """Test that parsed expression matches manually constructed Narwhals expression."""
    parsed = parse_filter_string("age > 25")
    expected = nw.col("age") > 25

    # Verify they produce the same results
    df = pl.DataFrame({"age": [20, 25, 30, 35]})
    lf = nw.from_native(df.lazy())

    parsed_result = lf.filter(parsed).collect().to_native()["age"].to_list()
    expected_result = lf.filter(expected).collect().to_native()["age"].to_list()

    assert parsed_result == expected_result == [30, 35]


def test_parsed_expression_equals_narwhals_expression_compound() -> None:
    """Test compound expression with AND operator."""
    parsed = parse_filter_string("age > 25 AND status = 'active'")
    expected = (nw.col("age") > 25) & (nw.col("status") == "active")

    df = pl.DataFrame(
        {"age": [20, 30, 30, 40], "status": ["active", "active", "inactive", "active"]}
    )
    lf = nw.from_native(df.lazy())

    parsed_result = lf.filter(parsed).collect().to_native()["age"].to_list()
    expected_result = lf.filter(expected).collect().to_native()["age"].to_list()

    assert parsed_result == expected_result == [30, 40]


def test_parsed_expression_equals_narwhals_expression_or() -> None:
    """Test compound expression with OR operator."""
    parsed = parse_filter_string("age < 20 OR age > 30")
    expected = (nw.col("age") < 20) | (nw.col("age") > 30)

    df = pl.DataFrame({"age": [15, 20, 25, 30, 35]})
    lf = nw.from_native(df.lazy())

    parsed_result = lf.filter(parsed).collect().to_native()["age"].to_list()
    expected_result = lf.filter(expected).collect().to_native()["age"].to_list()

    assert parsed_result == expected_result == [15, 35]


def test_parsed_expression_equals_narwhals_expression_not() -> None:
    """Test NOT operator."""
    parsed = parse_filter_string("NOT is_active")
    expected = ~nw.col("is_active")

    df = pl.DataFrame({"is_active": [True, False, True, False]})
    lf = nw.from_native(df.lazy())

    parsed_result = lf.filter(parsed).collect().to_native()["is_active"].to_list()
    expected_result = lf.filter(expected).collect().to_native()["is_active"].to_list()

    assert parsed_result == expected_result == [False, False]


def test_parsed_expression_equals_narwhals_expression_null() -> None:
    """Test NULL comparison with IS NULL."""
    parsed = parse_filter_string("deleted_at = NULL")
    expected = nw.col("deleted_at").is_null()

    df = pl.DataFrame({"deleted_at": [None, "2024-01-01", None, "2024-02-01"]})
    lf = nw.from_native(df.lazy())

    parsed_result = (
        lf.filter(parsed).collect().to_native()["deleted_at"].is_null().to_list()
    )
    expected_result = (
        lf.filter(expected).collect().to_native()["deleted_at"].is_null().to_list()
    )

    assert parsed_result == expected_result == [True, True]


@pytest.mark.parametrize(
    ("filter_string", "expected_expr", "expected_values"),
    [
        ("x = 5", nw.col("x") == 5, [5]),
        ("x != 5", nw.col("x") != 5, [3, 7, 10]),
        ("x > 5", nw.col("x") > 5, [7, 10]),
        ("x < 5", nw.col("x") < 5, [3]),
        ("x >= 5", nw.col("x") >= 5, [5, 7, 10]),
        ("x <= 5", nw.col("x") <= 5, [3, 5]),
    ],
)
def test_parsed_expression_equals_narwhals_expression_all_comparisons(
    filter_string: str,
    expected_expr: nw.Expr,
    expected_values: list[int],
) -> None:
    """Test that every comparison operator produces equivalent Narwhals expressions."""
    df = pl.DataFrame({"x": [3, 5, 7, 10]})
    lf = nw.from_native(df.lazy())

    parsed = parse_filter_string(filter_string)

    parsed_result = lf.filter(parsed).collect().to_native()["x"].to_list()
    expected_result = lf.filter(expected_expr).collect().to_native()["x"].to_list()

    assert parsed_result == expected_result == expected_values


@pytest.mark.parametrize(
    ("filter_string", "expected_expr"),
    [
        (
            "((age >= 25 AND (status = 'active' OR status = 'inactive')) "
            "OR NOT is_active) AND deleted_at = NULL",
            (
                (
                    (nw.col("age") >= nw.lit(25))
                    & (
                        (nw.col("status") == nw.lit("active"))
                        | (nw.col("status") == nw.lit("inactive"))
                    )
                )
                | (~nw.col("is_active"))
            )
            & nw.col("deleted_at").is_null(),
        ),
        (
            "NOT (status = 'deleted' OR deleted_at != NULL) "
            "AND (age < 30 OR NOT is_active)",
            (
                ~(
                    (nw.col("status") == nw.lit("deleted"))
                    | (~nw.col("deleted_at").is_null())
                )
            )
            & ((nw.col("age") < nw.lit(30)) | (~nw.col("is_active"))),
        ),
    ],
)
def test_parse_deeply_nested_expressions(
    filter_string: str,
    expected_expr: nw.Expr,
) -> None:
    """Ensure parsing handles nested parentheses and mixed operators."""
    expr = parse_filter_string(filter_string)
    assert repr(expr) == repr(expected_expr)


def test_feature_dep_filters_parsing(sample_lazy_frame: nw.LazyFrame[Any]) -> None:
    dep = FeatureDep(feature="upstream", filters=["age >= 25", "status = 'active'"])
    assert dep.filters is not None  # Type guard for basedpyright
    filtered_ids = (
        sample_lazy_frame.filter(*dep.filters).collect().to_native().sort("id")["id"]
    )
    assert filtered_ids.to_list() == [2]


def test_feature_dep_filters_invalid() -> None:
    dep = FeatureDep(feature="upstream", filters=["age BETWEEN 20 AND 30"])
    # Error is raised when accessing filters property (lazy evaluation via @cached_property)
    with pytest.raises(FilterParseError):
        _ = dep.filters
