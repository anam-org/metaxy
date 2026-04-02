"""Deprecated: use metaxy.ext.postgresql.metadata_store instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.postgresql import PostgreSQLMetadataStore, PostgreSQLMetadataStoreConfig

_warn_deprecated_module("metaxy.ext.metadata_stores.postgresql", "metaxy.ext.postgresql")

__all__ = ["PostgreSQLMetadataStore", "PostgreSQLMetadataStoreConfig"]
