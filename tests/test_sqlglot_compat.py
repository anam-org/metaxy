"""Tests for Metaxy's sqlglot compatibility patch."""

from typing import Any

from sqlglot.expressions import column

import metaxy._compat.sqlglot  # noqa: F401  # Ensure patch executes


def test_sqlglot_alias_accepts_bytes() -> None:
    """Ensure byte aliases are decoded rather than raising ValueError."""
    bytes_alias: Any = b"bytes_alias"
    expr = column("a").as_(bytes_alias)
    assert "bytes_alias" in expr.sql()
