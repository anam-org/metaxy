"""SQL helper utilities for MetadataStore backends."""

from __future__ import annotations

import sqlglot
from sqlglot import exp


def predicate_from_select_sql(select_sql: str) -> str:
    """Extract a WHERE predicate from a SELECT statement.

    Strips table qualifiers so the predicate can be reused elsewhere.
    If no WHERE clause is present, returns a tautology (TRUE).
    """
    parsed = sqlglot.parse_one(select_sql)

    where_expr = parsed.args.get("where")
    predicate = where_expr.this if where_expr is not None else exp.true()

    def _strip_table(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and node.args.get("table") is not None:
            cleaned = node.copy()
            cleaned.set("table", None)
            return cleaned
        return node

    normalized = predicate.transform(_strip_table)
    return normalized.sql()


def validate_identifier(identifier: str, context: str = "identifier") -> None:
    """Validate that a SQL identifier is safe."""
    if not identifier:
        raise ValueError(f"Empty {context} not allowed")

    if not all(c.isalnum() or c in ("_", ".") for c in identifier):
        raise ValueError(
            f"Invalid {context}: '{identifier}'. Only alphanumeric, underscore, and dot allowed."
        )

    if identifier[0].isdigit():
        raise ValueError(f"Invalid {context}: '{identifier}'. Cannot start with digit.")
