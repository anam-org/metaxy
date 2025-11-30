"""Centralized SQL utilities for safe value formatting and escaping.

This module provides utilities for safely formatting SQL values without SQL injection risks.
Prefer parameterized queries when possible; use these only when parameterization isn't supported.
"""

from datetime import datetime
from typing import Any

import sqlglot
from sqlglot import exp


class SQLValueFormatter:
    """Format Python values for SQL predicates with proper escaping.

    This is a last resort for backends that don't support parameterized queries.
    Always prefer driver-level parameterization when available.
    """

    MAX_PREDICATE_ROWS = 1000
    """Maximum number of rows to expand into OR predicates before warning/error."""

    @staticmethod
    def format_value(value: Any, dialect: str = "standard") -> str:
        """Format a Python value as a SQL literal with proper escaping.

        Args:
            value: Python value to format
            dialect: SQL dialect ('standard', 'clickhouse', 'delta', 'lancedb')

        Returns:
            SQL literal string

        Raises:
            ValueError: If value type is not supported
        """
        if value is None:
            return "NULL"

        if isinstance(value, bool):
            # Must check bool before int since bool is subclass of int
            if dialect == "clickhouse":
                return "1" if value else "0"
            return "TRUE" if value else "FALSE"

        if isinstance(value, int):
            # Integers are safe - no escaping needed
            return str(value)

        if isinstance(value, float):
            # Floats are safe - no escaping needed
            # Use repr to preserve precision
            return repr(value)

        if isinstance(value, str):
            # Escape single quotes by doubling them (SQL standard)
            escaped = value.replace("'", "''")
            return f"'{escaped}'"

        if isinstance(value, datetime):
            # Format as ISO string
            iso_str = value.isoformat()
            if dialect == "delta":
                return f"CAST('{iso_str}' AS TIMESTAMP)"
            if dialect == "lancedb":
                return f"TIMESTAMP '{iso_str}'"
            if dialect == "clickhouse":
                return f"'{iso_str}'"
            # Standard SQL
            return f"TIMESTAMP '{iso_str}'"

        # Reject unsupported types to prevent SQL injection
        # Callers should convert values to supported types before formatting
        raise ValueError(
            f"Unsupported value type for SQL formatting: {type(value).__name__}. "
            f"Supported types: None, bool, int, float, str, datetime. "
            f"Please convert the value to a supported type before formatting."
        )

    @staticmethod
    def format_predicate_condition(
        column: str, value: Any, dialect: str = "standard"
    ) -> str:
        """Format a single column=value condition.

        Args:
            column: Column name (should be validated/sanitized by caller)
            value: Value to compare
            dialect: SQL dialect

        Returns:
            SQL condition string like "col = 'value'" or "col IS NULL"
        """
        if value is None:
            return f"{column} IS NULL"
        formatted_value = SQLValueFormatter.format_value(value, dialect)
        return f"{column} = {formatted_value}"

    @staticmethod
    def format_row_predicate(
        row: dict[str, Any], columns: list[str], dialect: str = "standard"
    ) -> str:
        """Format a single row as an AND predicate.

        Args:
            row: Dictionary mapping column names to values
            columns: List of column names to include
            dialect: SQL dialect

        Returns:
            SQL predicate like "(col1 = 'val1' AND col2 = 42)"
        """
        conditions = [
            SQLValueFormatter.format_predicate_condition(col, row[col], dialect)
            for col in columns
        ]
        if len(conditions) == 1:
            return conditions[0]
        return f"({' AND '.join(conditions)})"

    @staticmethod
    def format_multiple_rows_predicate(
        rows: list[dict[str, Any]],
        columns: list[str],
        dialect: str = "standard",
        max_rows: int | None = None,
    ) -> str:
        """Format multiple rows as an OR predicate.

        Args:
            rows: List of row dictionaries
            columns: List of column names to include in predicate
            dialect: SQL dialect
            max_rows: Maximum number of rows to process (default: MAX_PREDICATE_ROWS)

        Returns:
            SQL predicate like "(col1 = 'a' AND col2 = 1) OR (col1 = 'b' AND col2 = 2)"

        Raises:
            ValueError: If number of rows exceeds max_rows
        """
        if max_rows is None:
            max_rows = SQLValueFormatter.MAX_PREDICATE_ROWS

        if len(rows) > max_rows:
            raise ValueError(
                f"Cannot expand {len(rows)} rows into OR predicate. "
                f"Maximum allowed: {max_rows}. "
                f"Consider using a more selective filter or implementing chunked operations."
            )

        if not rows:
            raise ValueError("Cannot create predicate from empty row list")

        predicates = [
            SQLValueFormatter.format_row_predicate(row, columns, dialect)
            for row in rows
        ]

        if len(predicates) == 1:
            return predicates[0]

        return " OR ".join(predicates)

    @staticmethod
    def format_update_assignments(
        updates: dict[str, Any], dialect: str = "standard"
    ) -> str:
        """Format UPDATE SET clause assignments.

        Args:
            updates: Dictionary mapping column names to new values
            dialect: SQL dialect

        Returns:
            SQL SET clause like "col1 = 'val1', col2 = 42"
        """
        assignments = []
        for col, value in updates.items():
            if value is None:
                assignments.append(f"{col} = NULL")
            else:
                formatted_value = SQLValueFormatter.format_value(value, dialect)
                assignments.append(f"{col} = {formatted_value}")

        return ", ".join(assignments)


def predicate_from_select_sql(select_sql: str) -> str:
    """Extract a WHERE predicate from a SELECT statement.

    We strip table qualifiers/aliases from column references so the predicate can be
    re-used in other statements (DELETE/COUNT) without requiring the same aliases.
    If no WHERE clause is present, returns a tautology predicate.
    """
    parsed = sqlglot.parse_one(select_sql)

    where_expr = parsed.args.get("where")
    predicate = where_expr.this if where_expr is not None else exp.true()

    # Remove table qualifiers to avoid alias dependencies
    def _strip_table(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and node.args.get("table") is not None:
            cleaned = node.copy()
            cleaned.set("table", None)
            return cleaned
        return node

    normalized = predicate.transform(_strip_table)
    return normalized.sql()


def build_in_predicate_from_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    dialect: str = "standard",
    max_rows: int | None = None,
) -> str:
    """Compact IN/OR predicate builder shared by file/embedded backends.

    Uses `IN` for single-column filters to avoid large OR chains and falls back to
    SQLValueFormatter for multi-column cases.
    """
    if not columns:
        raise ValueError("No columns provided for predicate construction")

    if len(columns) == 1:
        # Single-column IN clause keeps predicates compact and pushdown-friendly
        col = columns[0]
        if max_rows is None:
            max_rows = SQLValueFormatter.MAX_PREDICATE_ROWS
        if len(rows) > max_rows:
            raise ValueError(
                f"Cannot expand {len(rows)} rows into IN predicate. "
                f"Maximum allowed: {max_rows}. "
                "Consider chunking the operation or using a more selective filter."
            )
        formatted_values = [
            SQLValueFormatter.format_value(r[col], dialect) for r in rows
        ]
        values_sql = ", ".join(formatted_values)
        return f"{col} IN ({values_sql})"

    # Multi-column predicates fall back to AND/OR expansion with bounds checking
    return SQLValueFormatter.format_multiple_rows_predicate(
        rows=rows,
        columns=columns,
        dialect=dialect,
        max_rows=max_rows,
    )


def validate_identifier(identifier: str, context: str = "identifier") -> None:
    """Validate that a SQL identifier is safe.

    Args:
        identifier: SQL identifier (table/column name)
        context: Context string for error messages

    Raises:
        ValueError: If identifier contains unsafe characters
    """
    # Basic validation: alphanumeric, underscores, and dots (for qualified names)
    # This is intentionally strict to prevent SQL injection
    if not identifier:
        raise ValueError(f"Empty {context} not allowed")

    # Allow alphanumeric, underscore, and dot
    if not all(c.isalnum() or c in ("_", ".") for c in identifier):
        raise ValueError(
            f"Invalid {context}: '{identifier}'. "
            f"Only alphanumeric characters, underscores, and dots allowed."
        )

    # Don't allow starting with a digit
    if identifier[0].isdigit():
        raise ValueError(f"Invalid {context}: '{identifier}'. Cannot start with digit.")
