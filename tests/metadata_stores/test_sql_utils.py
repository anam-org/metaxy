"""Tests for centralized SQL utilities."""

from datetime import datetime

import pytest

from metaxy.metadata_store._sql_utils import (
    SQLValueFormatter,
    validate_identifier,
)


class TestSQLValueFormatter:
    """Tests for SQLValueFormatter."""

    def test_format_value_none(self):
        """Test NULL formatting."""
        assert SQLValueFormatter.format_value(None) == "NULL"

    def test_format_value_bool(self):
        """Test boolean formatting."""
        assert SQLValueFormatter.format_value(True) == "TRUE"
        assert SQLValueFormatter.format_value(False) == "FALSE"
        # ClickHouse dialect uses 0/1
        assert SQLValueFormatter.format_value(True, "clickhouse") == "1"
        assert SQLValueFormatter.format_value(False, "clickhouse") == "0"

    def test_format_value_int(self):
        """Test integer formatting."""
        assert SQLValueFormatter.format_value(42) == "42"
        assert SQLValueFormatter.format_value(0) == "0"
        assert SQLValueFormatter.format_value(-123) == "-123"

    def test_format_value_float(self):
        """Test float formatting."""
        assert SQLValueFormatter.format_value(3.14) == "3.14"
        assert SQLValueFormatter.format_value(0.0) == "0.0"

    def test_format_value_string(self):
        """Test string formatting with escaping."""
        assert SQLValueFormatter.format_value("hello") == "'hello'"
        assert SQLValueFormatter.format_value("it's") == "'it''s'"
        assert SQLValueFormatter.format_value("a'b'c") == "'a''b''c'"

    def test_format_value_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2024, 1, 15, 10, 30, 0)

        # Standard SQL
        result = SQLValueFormatter.format_value(dt, "standard")
        assert "TIMESTAMP" in result
        assert "2024-01-15" in result

        # Delta
        result = SQLValueFormatter.format_value(dt, "delta")
        assert "CAST" in result
        assert "2024-01-15" in result

        # LanceDB
        result = SQLValueFormatter.format_value(dt, "lancedb")
        assert "TIMESTAMP" in result
        assert "2024-01-15" in result

    def test_format_predicate_condition(self):
        """Test single condition formatting."""
        assert SQLValueFormatter.format_predicate_condition("col", 42) == "col = 42"
        assert (
            SQLValueFormatter.format_predicate_condition("col", "hello")
            == "col = 'hello'"
        )
        assert (
            SQLValueFormatter.format_predicate_condition("col", None) == "col IS NULL"
        )

    def test_format_row_predicate(self):
        """Test row predicate formatting."""
        row = {"id": 1, "name": "test"}
        result = SQLValueFormatter.format_row_predicate(row, ["id", "name"])
        assert "id = 1" in result
        assert "name = 'test'" in result
        assert "AND" in result

        # Single column - no parens needed
        result = SQLValueFormatter.format_row_predicate(row, ["id"])
        assert result == "id = 1"

    def test_format_multiple_rows_predicate(self):
        """Test multiple rows OR predicate."""
        rows = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
        ]
        result = SQLValueFormatter.format_multiple_rows_predicate(rows, ["id", "name"])
        assert "id = 1" in result
        assert "name = 'a'" in result
        assert "id = 2" in result
        assert "name = 'b'" in result
        assert "OR" in result

    def test_format_multiple_rows_predicate_bounds(self):
        """Test bounds checking for large predicates."""
        # Create a large list of rows
        rows = [{"id": i} for i in range(2000)]

        # Should raise ValueError for too many rows
        with pytest.raises(ValueError, match="Cannot expand 2000 rows"):
            SQLValueFormatter.format_multiple_rows_predicate(
                rows, ["id"], max_rows=1000
            )

    def test_format_multiple_rows_predicate_empty(self):
        """Test empty row list raises error."""
        with pytest.raises(ValueError, match="Cannot create predicate from empty"):
            SQLValueFormatter.format_multiple_rows_predicate([], ["id"])

    def test_format_update_assignments(self):
        """Test UPDATE SET clause formatting."""
        updates = {"name": "test", "age": 25, "active": True}
        result = SQLValueFormatter.format_update_assignments(updates)
        assert "name = 'test'" in result
        assert "age = 25" in result
        assert "active = TRUE" in result

        # ClickHouse dialect for booleans
        result = SQLValueFormatter.format_update_assignments(updates, "clickhouse")
        assert "active = 1" in result

        # NULL values
        updates = {"name": None}
        result = SQLValueFormatter.format_update_assignments(updates)
        assert result == "name = NULL"


class TestValidateIdentifier:
    """Tests for identifier validation."""

    def test_valid_identifiers(self):
        """Test valid identifiers."""
        validate_identifier("table_name")
        validate_identifier("TableName")
        validate_identifier("table123")
        validate_identifier("table_name_123")
        validate_identifier("schema.table")

    def test_invalid_identifiers(self):
        """Test invalid identifiers."""
        # Empty
        with pytest.raises(ValueError, match="Empty"):
            validate_identifier("")

        # Starting with digit
        with pytest.raises(ValueError, match="Cannot start with digit"):
            validate_identifier("123table")

        # Special characters
        with pytest.raises(ValueError, match="Only alphanumeric"):
            validate_identifier("table-name")

        with pytest.raises(ValueError, match="Only alphanumeric"):
            validate_identifier("table name")

        with pytest.raises(ValueError, match="Only alphanumeric"):
            validate_identifier("table;DROP")
