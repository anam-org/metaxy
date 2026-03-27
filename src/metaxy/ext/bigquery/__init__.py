"""BigQuery metadata store — thin wrapper that composes BigQueryEngine + IbisStorageConfig."""

from metaxy.ext.bigquery.config import BigQueryMetadataStoreConfig
from metaxy.ext.bigquery.engine import BigQueryEngine
from metaxy.ext.bigquery.legacy_store import BigQueryMetadataStore

__all__ = [
    "BigQueryEngine",
    "BigQueryMetadataStore",
    "BigQueryMetadataStoreConfig",
]
