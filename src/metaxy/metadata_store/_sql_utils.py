"""SQL helper utilities for MetadataStore backends."""

from __future__ import annotations

from sqlglot import exp

from metaxy.metadata_store.utils import (
    _extract_where_expression,
    _strip_table_qualifiers,
)


def predicate_from_select_sql(select_sql: str) -> str:
    """Extract a WHERE predicate from a SELECT statement.

    Strips table qualifiers so the predicate can be reused elsewhere.
    If no WHERE clause is present, returns a tautology (TRUE).
    """
    predicate = _extract_where_expression(select_sql) or exp.true()
    normalized = predicate.transform(_strip_table_qualifiers())
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
