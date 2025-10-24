"""Demonstration of configuring the DuckLake metadata store.

This example mirrors the Dagster DuckLake resource behaviour while staying
Narwhals-compatible. No DuckLake installation is required to run this script;
we preview the SQL statements that would be executed when attaching DuckLake.
"""

from __future__ import annotations

from pathlib import Path

from metaxy import MetaxyConfig
from metaxy.metadata_store.ducklake import DuckLakeMetadataStore


def build_store() -> DuckLakeMetadataStore:
    """Create a DuckLakeMetadataStore using metaxy.toml configuration."""
    config_path = Path(__file__).with_name("metaxy.toml")
    config = MetaxyConfig.load(config_path, search_parents=False)
    store = config.get_store()
    if not isinstance(store, DuckLakeMetadataStore):
        raise RuntimeError(
            "DuckLake example misconfigured: expected DuckLakeMetadataStore."
        )
    return store


def preview_attachment_sql(store: DuckLakeMetadataStore) -> list[str]:
    """Return the SQL statements DuckLake would execute on open()."""
    return store.preview_attachment_sql()


if __name__ == "__main__":
    ducklake_store = build_store()
    print("DuckLake store initialised. Extensions:", ducklake_store.extensions)
    print("\nPreview of DuckLake ATTACH SQL:")
    for line in preview_attachment_sql(ducklake_store):
        print("  ", line)
