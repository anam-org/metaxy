"""Tests for table metadata utilities in Dagster integration."""

from __future__ import annotations

from typing import Any

import dagster as dg
import narwhals as nw
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy.ext.dagster.table_metadata import build_table_preview_metadata


@pytest.fixture
def basic_schema() -> dg.TableSchema:
    """Simple schema with common primitive types."""
    return dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="name", type="str"),
            dg.TableColumn(name="value", type="float"),
            dg.TableColumn(name="active", type="bool"),
        ]
    )


def test_build_table_preview_basic_types(basic_schema: dg.TableSchema, snapshot: SnapshotAssertion) -> None:
    """Test basic functionality with primitive types (int, str, float, bool).

    Primitive types should be preserved as-is in the table records.
    """
    # Create a DataFrame with primitive types
    df = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [1.5, 2.7, 3.9],
            "active": [True, False, True],
        }
    )
    lazy_df = nw.from_native(df)

    # Build table preview
    result = build_table_preview_metadata(lazy_df, basic_schema, n_rows=5)

    # Verify metadata value type and structure
    assert isinstance(result, dg.TableMetadataValue)
    assert result.schema == basic_schema
    assert len(result.records) == 3

    # Snapshot the records
    records_data = [record.data for record in result.records]
    assert records_data == snapshot


def test_build_table_preview_empty_dataframe(
    basic_schema: dg.TableSchema,
) -> None:
    """Test that empty DataFrame returns empty records list with schema."""
    # Create an empty DataFrame with the same schema
    df = pl.LazyFrame(
        schema={
            "id": pl.Int64,
            "name": pl.String,
            "value": pl.Float64,
            "active": pl.Boolean,
        }
    )
    lazy_df = nw.from_native(df)

    # Build table preview
    result = build_table_preview_metadata(lazy_df, basic_schema, n_rows=5)

    # Verify empty results
    assert isinstance(result, dg.TableMetadataValue)
    assert result.schema == basic_schema
    assert len(result.records) == 0


def test_build_table_preview_fewer_rows_than_n_rows(basic_schema: dg.TableSchema, snapshot: SnapshotAssertion) -> None:
    """Test DataFrame with fewer rows than n_rows returns all rows."""
    # Create a DataFrame with 3 rows
    df = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [1.5, 2.7, 3.9],
            "active": [True, False, True],
        }
    )
    lazy_df = nw.from_native(df)

    # Request more rows than available
    result = build_table_preview_metadata(lazy_df, basic_schema, n_rows=10)

    # Should return all 3 rows
    assert len(result.records) == 3
    records_data = [record.data for record in result.records]
    assert records_data == snapshot


def test_build_table_preview_more_rows_than_n_rows(basic_schema: dg.TableSchema, snapshot: SnapshotAssertion) -> None:
    """Test DataFrame with more rows than n_rows returns only last n_rows."""
    # Create a DataFrame with 10 rows
    df = pl.LazyFrame(
        {
            "id": list(range(1, 11)),
            "name": [f"Person{i}" for i in range(1, 11)],
            "value": [float(i) * 1.1 for i in range(1, 11)],
            "active": [i % 2 == 0 for i in range(1, 11)],
        }
    )
    lazy_df = nw.from_native(df)

    # Request only last 5 rows
    result = build_table_preview_metadata(lazy_df, basic_schema, n_rows=5)

    # Should return last 5 rows (rows 6-10)
    assert len(result.records) == 5
    records_data = [record.data for record in result.records]
    assert records_data == snapshot

    # Verify it's the last 5 rows by checking IDs
    ids = [record.data["id"] for record in result.records]
    assert ids == [6, 7, 8, 9, 10]


def test_build_table_preview_custom_n_rows(basic_schema: dg.TableSchema, snapshot: SnapshotAssertion) -> None:
    """Test custom n_rows parameter with different value."""
    # Create a DataFrame with 10 rows
    df = pl.LazyFrame(
        {
            "id": list(range(1, 11)),
            "name": [f"Person{i}" for i in range(1, 11)],
            "value": [float(i) * 1.1 for i in range(1, 11)],
            "active": [i % 2 == 0 for i in range(1, 11)],
        }
    )
    lazy_df = nw.from_native(df)

    # Request only last 3 rows
    result = build_table_preview_metadata(lazy_df, basic_schema, n_rows=3)

    # Should return last 3 rows (rows 8-10)
    assert len(result.records) == 3
    records_data = [record.data for record in result.records]
    assert records_data == snapshot

    # Verify it's the last 3 rows by checking IDs
    ids = [record.data["id"] for record in result.records]
    assert ids == [8, 9, 10]


def test_build_table_preview_struct_columns(snapshot: SnapshotAssertion) -> None:
    """Test that Struct columns are converted to JSON strings."""
    # Create a DataFrame with Struct column
    df = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "user": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )
    lazy_df = nw.from_native(df)

    # Create schema
    schema = dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="user", type="Struct"),
        ]
    )

    # Build table preview
    result = build_table_preview_metadata(lazy_df, schema, n_rows=5)

    # Verify struct columns are converted to JSON strings
    assert len(result.records) == 3
    records_data = [record.data for record in result.records]

    # Check that user field is a JSON string, not a dict
    for record_data in records_data:
        assert isinstance(record_data["user"], str)

    assert records_data == snapshot


def test_build_table_preview_list_columns(snapshot: SnapshotAssertion) -> None:
    """Test that List columns are converted to JSON strings."""
    # Create a DataFrame with List column
    df = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "tags": [["python", "data"], ["ml", "ai"], ["testing"]],
        }
    )
    lazy_df = nw.from_native(df)

    # Create schema
    schema = dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="tags", type="List"),
        ]
    )

    # Build table preview
    result = build_table_preview_metadata(lazy_df, schema, n_rows=5)

    # Verify list columns are converted to JSON strings
    assert len(result.records) == 3
    records_data = [record.data for record in result.records]

    # Check that tags field is a JSON string, not a list
    for record_data in records_data:
        assert isinstance(record_data["tags"], str)

    assert records_data == snapshot


def test_build_table_preview_array_columns(snapshot: SnapshotAssertion) -> None:
    """Test that Array columns are converted to JSON strings."""
    # Create a DataFrame with Array column (fixed-size list)
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "scores": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }
    ).with_columns(pl.col("scores").cast(pl.Array(pl.Float64, 3)))
    lazy_df = nw.from_native(df.lazy())

    # Create schema
    schema = dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="scores", type="Array"),
        ]
    )

    # Build table preview
    result = build_table_preview_metadata(lazy_df, schema, n_rows=5)

    # Verify array columns are converted to JSON strings
    assert len(result.records) == 3
    records_data = [record.data for record in result.records]

    # Check that scores field is a JSON string, not a list
    for record_data in records_data:
        assert isinstance(record_data["scores"], str)

    assert records_data == snapshot


def test_build_table_preview_null_values(snapshot: SnapshotAssertion) -> None:
    """Test that null values are handled properly (passed through as None)."""
    # Create a DataFrame with null values
    df = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", None, "Charlie"],
            "value": [1.5, None, 3.9],
            "active": [True, None, False],
        }
    )
    lazy_df = nw.from_native(df)

    # Create schema
    schema = dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="name", type="str"),
            dg.TableColumn(name="value", type="float"),
            dg.TableColumn(name="active", type="bool"),
        ]
    )

    # Build table preview
    result = build_table_preview_metadata(lazy_df, schema, n_rows=5)

    # Verify null values are preserved
    assert len(result.records) == 3
    records_data = [record.data for record in result.records]

    # Check that nulls are None
    assert records_data[1]["name"] is None
    assert records_data[1]["value"] is None
    assert records_data[1]["active"] is None

    assert records_data == snapshot


def test_build_table_preview_mixed_complex_and_primitive(
    snapshot: SnapshotAssertion,
) -> None:
    """Test DataFrame with both complex and primitive types together."""
    # Create a DataFrame with mix of types
    df = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "metadata": [
                {"key": "value1"},
                {"key": "value2"},
                {"key": "value3"},
            ],
            "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
            "score": [1.5, 2.7, 3.9],
        }
    )
    lazy_df = nw.from_native(df)

    # Create schema
    schema = dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="name", type="str"),
            dg.TableColumn(name="metadata", type="Struct"),
            dg.TableColumn(name="tags", type="List"),
            dg.TableColumn(name="score", type="float"),
        ]
    )

    # Build table preview
    result = build_table_preview_metadata(lazy_df, schema, n_rows=5)

    # Verify mixed types are handled correctly
    assert len(result.records) == 3
    records_data = [record.data for record in result.records]

    # Primitive types should be preserved
    for i, record_data in enumerate(records_data):
        assert isinstance(record_data["id"], int)
        assert isinstance(record_data["name"], str)
        assert isinstance(record_data["score"], float)

        # Complex types should be JSON strings
        assert isinstance(record_data["metadata"], str)
        assert isinstance(record_data["tags"], str)

    assert records_data == snapshot


def test_build_table_preview_with_narwhals_lazyframe() -> None:
    """Test that function works with narwhals LazyFrame (not just polars)."""
    # Create a polars DataFrame, wrap it with narwhals
    df_polars = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    lazy_df_nw: nw.LazyFrame[Any] = nw.from_native(df_polars)

    schema = dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="value", type="int"),
        ]
    )

    # Should work without errors
    result = build_table_preview_metadata(lazy_df_nw, schema, n_rows=5)

    assert len(result.records) == 3
    assert result.records[0].data["id"] == 1
    assert result.records[0].data["value"] == 10


def test_build_table_preview_list_truncation() -> None:
    """Test that long lists are truncated to show first 1 and last 1 items."""
    # Create a DataFrame with long list column
    df = pl.LazyFrame(
        {
            "id": [1, 2, 3],
            "short_list": [[1, 2], [4], [6]],  # <= 2 items, no truncation
            "long_list": [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 10 items -> truncated
                [10, 20, 30, 40, 50],  # 5 items -> truncated
                [1, 2, 3],  # 3 items -> truncated
            ],
        }
    )
    lazy_df = nw.from_native(df)

    schema = dg.TableSchema(
        columns=[
            dg.TableColumn(name="id", type="int"),
            dg.TableColumn(name="short_list", type="List"),
            dg.TableColumn(name="long_list", type="List"),
        ]
    )

    result = build_table_preview_metadata(lazy_df, schema, n_rows=5)

    assert len(result.records) == 3
    records_data = [record.data for record in result.records]

    # Short lists (<= 2 items) should not be truncated
    assert records_data[0]["short_list"] == "[1,2]"
    assert records_data[1]["short_list"] == "[4]"
    assert records_data[2]["short_list"] == "[6]"

    # Long lists (> 2 items) should be truncated to show first 1 and last 1 with ..
    assert records_data[0]["long_list"] == "[1,..,10]"
    assert records_data[1]["long_list"] == "[10,..,50]"
    assert records_data[2]["long_list"] == "[1,..,3]"
