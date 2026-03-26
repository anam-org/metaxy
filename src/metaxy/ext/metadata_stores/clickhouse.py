"""Backwards-compatibility shim — imports from metaxy.ext.clickhouse."""

from metaxy.ext.clickhouse import (
    ClickHouseEngine,
    ClickHouseMetadataStore,
    ClickHouseMetadataStoreConfig,
    ClickHouseSQLHandler,
    ClickHouseVersioningEngine,
)

__all__ = [
    "ClickHouseEngine",
    "ClickHouseMetadataStore",
    "ClickHouseMetadataStoreConfig",
    "ClickHouseSQLHandler",
    "ClickHouseVersioningEngine",
]
