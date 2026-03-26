"""PostgreSQL metadata store configuration."""

from __future__ import annotations

from pydantic import Field

from metaxy._decorators import experimental, public
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig


@public
@experimental
class PostgreSQLMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for PostgreSQLMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from IbisMetadataStoreConfig.
    """

    auto_cast_struct_for_jsonb: bool = Field(
        default=True,
        description="Whether to encode/decode Struct columns to/from JSON on writes/reads. Metaxy system columns are always converted.",
    )
