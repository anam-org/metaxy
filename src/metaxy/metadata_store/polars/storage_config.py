"""Storage configuration for Polars-backed handlers."""

from __future__ import annotations

from metaxy.metadata_store.storage_config import StorageConfig


class PolarsStorageConfig(StorageConfig):
    """Storage configuration for Polars-backed handlers (Delta, Iceberg, LanceDB)."""
