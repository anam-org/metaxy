"""Demonstration of configuring the DuckLake metadata store.

This script loads the store from `metaxy.toml` and prints the SQL statements
that would be executed when attaching DuckLake.
"""

import metaxy as mx
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore


def preview_attachment_sql(store: DuckDBMetadataStore) -> list[str]:
    """Return the SQL statements DuckLake would execute on open()."""
    return store.preview_ducklake_sql()


if __name__ == "__main__":
    # Initialize Metaxy and load config from metaxy.toml
    config = mx.init()
    ducklake_store = config.get_store()
    assert isinstance(ducklake_store, DuckDBMetadataStore), (
        "DuckLake example misconfigured: expected DuckDBMetadataStore."
    )
    ducklake_config = ducklake_store.ducklake_attachment_config

    print("DuckLake store initialised")
    print("  Store class:", ducklake_store.__class__.__name__)
    print("  Database:", ducklake_store.database)
    print("  Catalog backend:", ducklake_config.catalog.type)
    storage_backend = (
        ducklake_config.storage.type
        if ducklake_config.storage is not None
        else "motherduck-managed"
    )
    print("  Storage backend:", storage_backend)
    print()
    print("Preview of DuckLake setup SQL:")
    for idx, line in enumerate(preview_attachment_sql(ducklake_store), start=1):
        print(f"  {idx:02d}. {line}")

    print()
    print("Running SHOW ALL TABLES after opening the store:")
    with ducklake_store.open("w"):
        rows = (
            ducklake_store._duckdb_raw_connection()
            .execute("SHOW ALL TABLES")
            .fetchall()
        )

    print(f"  Found {len(rows)} tables")
    if rows:
        # SHOW ALL TABLES returns rich metadata per table; print table names only.
        table_names = [str(row[2]) for row in rows]
        preview_count = min(8, len(table_names))
        for table_name in table_names[:preview_count]:
            print(f"   - {table_name}")
        if len(table_names) > preview_count:
            print(f"   ... and {len(table_names) - preview_count} more")
