"""Common fixtures for Rivers integration tests."""

from collections.abc import Iterator

import metaxy as mx
import pytest


@pytest.fixture
def metaxy_config() -> Iterator[mx.MetaxyConfig]:
    """In-memory DuckDB-backed Metaxy config, scoped to a single test."""
    store_config = mx.StoreConfig(
        type="metaxy.ext.duckdb.DuckDBMetadataStore",
        config={"database": ":memory:"},
    )
    with mx.MetaxyConfig(project="test", stores={"dev": store_config}).use() as config:
        yield config