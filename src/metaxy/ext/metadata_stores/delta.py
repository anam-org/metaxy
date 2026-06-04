"""Deprecated: use metaxy.ext.polars.handlers.delta instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.polars.handlers.delta import DeltaMetadataStore, DeltaMetadataStoreConfig

_warn_deprecated_module("metaxy.ext.metadata_stores.delta", "metaxy.ext.polars.handlers.delta")

__all__ = ["DeltaMetadataStore", "DeltaMetadataStoreConfig"]
