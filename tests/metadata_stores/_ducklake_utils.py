"""Test helpers for DuckLake-enabled scenarios."""

from __future__ import annotations

import duckdb
import pytest


def ensure_ducklake_extension() -> None:
    """Ensure the DuckLake DuckDB extension is available, or skip tests."""
    conn = duckdb.connect()
    try:
        conn.execute("INSTALL ducklake;")
        conn.execute("LOAD ducklake;")
    except duckdb.Error as exc:
        pytest.skip(f"DuckLake extension unavailable: {exc}")
    finally:
        conn.close()
