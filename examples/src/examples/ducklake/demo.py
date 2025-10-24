"""Demonstration of configuring the DuckLake metadata store.

This example mirrors the Dagster DuckLake resource behaviour while staying
Narwhals-compatible. No DuckLake installation is required to run this script;
we preview the SQL statements that would be executed when attaching DuckLake.
"""

from __future__ import annotations

from metaxy.metadata_store.ducklake import (
    DuckLakeMetadataStore,
    DuckLakePostgresMetadataBackend,
    DuckLakeS3StorageBackend,
)


def build_store() -> DuckLakeMetadataStore:
    """Create a DuckLakeMetadataStore using typed configuration models."""
    metadata_backend = DuckLakePostgresMetadataBackend(
        database="ducklake_meta",
        user="ducklake",
        password="secret",
    )
    storage_backend = DuckLakeS3StorageBackend(
        endpoint_url="https://object-store",
        bucket="ducklake",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        prefix="metadata",
    )

    return DuckLakeMetadataStore(
        metadata_backend=metadata_backend,
        storage_backend=storage_backend,
        attach_options={"api_version": "0.2", "override_data_path": True},
        database=":memory:",
    )


def preview_attachment_sql(store: DuckLakeMetadataStore) -> list[str]:
    """Return the SQL statements DuckLake would execute on open()."""
    return store.preview_attachment_sql()


if __name__ == "__main__":
    ducklake_store = build_store()
    print("DuckLake store initialised. Extensions:", ducklake_store.extensions)
    print("\nPreview of DuckLake ATTACH SQL:")
    for line in preview_attachment_sql(ducklake_store):
        print("  ", line)
