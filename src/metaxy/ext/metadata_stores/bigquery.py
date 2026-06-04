"""Deprecated: use metaxy.ext.bigquery.metadata_store instead."""

from metaxy._warnings import _warn_deprecated_module
from metaxy.ext.bigquery import BigQueryMetadataStore, BigQueryMetadataStoreConfig

_warn_deprecated_module("metaxy.ext.metadata_stores.bigquery", "metaxy.ext.bigquery")

__all__ = ["BigQueryMetadataStore", "BigQueryMetadataStoreConfig"]
