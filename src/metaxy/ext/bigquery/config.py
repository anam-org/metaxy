"""BigQuery metadata store configuration."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig


@public
class BigQueryMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for BigQueryMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.bigquery.BigQueryMetadataStore"

        [stores.dev.config]
        project_id = "my-project"
        dataset_id = "my_dataset"
        credentials_path = "/path/to/service-account.json"
        ```
    """

    project_id: str | None = Field(default=None, description="Google Cloud project ID containing the dataset.")
    dataset_id: str | None = Field(default=None, description="BigQuery dataset name for storing metadata tables.")
    credentials_path: str | None = Field(default=None, description="Path to service account JSON file.")
    credentials: Any | None = Field(default=None, description="Google Cloud credentials object.")
    location: str | None = Field(
        default=None,
        description="Default location for BigQuery resources (e.g., 'US', 'EU').",
    )
