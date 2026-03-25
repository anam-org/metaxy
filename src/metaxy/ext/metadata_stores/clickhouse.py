"""Deprecated: use metaxy.ext.clickhouse.metadata_store instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.clickhouse.engine import ClickHouseVersioningEngine
from metaxy.ext.clickhouse.metadata_store import (
    ClickHouseMetadataStore,
    ClickHouseMetadataStoreConfig,
)

_warn_deprecated_module("metaxy.ext.metadata_stores.clickhouse", "metaxy.ext.clickhouse")

__all__ = ["ClickHouseMetadataStore", "ClickHouseMetadataStoreConfig", "ClickHouseVersioningEngine"]
