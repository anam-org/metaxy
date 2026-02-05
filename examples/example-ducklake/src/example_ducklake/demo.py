"""Demonstration of configuring the DuckLake metadata store.

This example mirrors the Dagster DuckLake resource behaviour while staying
Narwhals-compatible. No DuckLake installation is required to run this script;
we preview the SQL statements that would be executed when attaching DuckLake.
"""

import metaxy as mx
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore


def preview_attachment_sql(store: DuckDBMetadataStore) -> list[str]:
    """Return the SQL statements DuckLake would execute on open()."""
    return store.preview_ducklake_sql()


if __name__ == "__main__":
    # Initialize metaxy and get config (searches for metaxy.toml)
    config = mx.init_metaxy()
    ducklake_store = config.get_store()
    assert isinstance(ducklake_store, DuckDBMetadataStore), (
        "DuckLake example misconfigured: expected DuckDBMetadataStore."
    )
    ducklake_store.ducklake_attachment
    print("DuckLake store initialised. Extensions:", ducklake_store.extensions)
    print("\nPreview of DuckLake ATTACH SQL:")
    for line in preview_attachment_sql(ducklake_store):
        print("  ", line)
