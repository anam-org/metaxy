from __future__ import annotations

from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig


@public
class ClickHouseMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for ClickHouseMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.clickhouse.ClickHouseMetadataStore"

        [stores.dev.config]
        connection_string = "clickhouse://localhost:8443/default"
        hash_algorithm = "xxhash64"
        ```
    """

    auto_cast_struct_for_map: bool = Field(
        default=True,
        description="Auto-convert DataFrame Struct columns to Map format on write when the ClickHouse column is Map type. Metaxy system columns are always converted.",
    )
