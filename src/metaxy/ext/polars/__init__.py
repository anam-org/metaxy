"""Polars-based metadata store handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from metaxy._warnings import _warn_deprecated_module

if TYPE_CHECKING:
    from metaxy.ext.polars.handlers.delta import DeltaMetadataStore, DeltaMetadataStoreConfig
    from metaxy.ext.polars.handlers.iceberg import IcebergMetadataStore, IcebergMetadataStoreConfig, TableIdentifier

__all__ = [
    "DeltaMetadataStore",
    "DeltaMetadataStoreConfig",
    "IcebergMetadataStore",
    "IcebergMetadataStoreConfig",
    "TableIdentifier",
]

_DELTA_ATTRS = {"DeltaMetadataStore", "DeltaMetadataStoreConfig"}
_ICEBERG_ATTRS = {"IcebergMetadataStore", "IcebergMetadataStoreConfig", "TableIdentifier"}


def __getattr__(name: str) -> Any:
    if name in _DELTA_ATTRS:
        from metaxy.ext.polars.handlers.delta import DeltaMetadataStore, DeltaMetadataStoreConfig

        globals()["DeltaMetadataStore"] = DeltaMetadataStore
        globals()["DeltaMetadataStoreConfig"] = DeltaMetadataStoreConfig

        _warn_deprecated_module(
            "metaxy.ext.polars.DeltaMetadataStore",
            "metaxy.ext.polars.handlers.delta.DeltaMetadataStore",
        )

        return globals()[name]
    if name in _ICEBERG_ATTRS:
        from metaxy.ext.polars.handlers.iceberg import (
            IcebergMetadataStore,
            IcebergMetadataStoreConfig,
            TableIdentifier,
        )

        globals()["IcebergMetadataStore"] = IcebergMetadataStore
        globals()["IcebergMetadataStoreConfig"] = IcebergMetadataStoreConfig
        globals()["TableIdentifier"] = TableIdentifier

        _warn_deprecated_module(
            "metaxy.ext.polars.IcebergMetadataStore",
            "metaxy.ext.polars.handlers.iceberg.IcebergMetadataStore",
        )

        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
