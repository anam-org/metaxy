"""Demonstration of configuring the DuckLake metadata store.

This example mirrors the Dagster DuckLake resource behaviour while staying
Narwhals-compatible. No DuckLake installation is required to run this script;
we preview the SQL statements that would be executed when attaching DuckLake.
"""

from __future__ import annotations

from metaxy.metadata_store.ducklake import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
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


class PreviewCursor:
    """Captures SQL commands for display."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, command: str) -> None:
        self.commands.append(command.strip())

    def close(self) -> None:  # pragma: no cover - harmless in example
        pass


class PreviewConnection:
    """Minimal DuckDB-like connection for previewing attachment SQL."""

    def __init__(self) -> None:
        self._cursor = PreviewCursor()

    def cursor(self) -> PreviewCursor:
        return self._cursor


def preview_attachment_sql(store: DuckLakeMetadataStore) -> list[str]:
    """Return the SQL statements DuckLake would execute on open()."""
    attachment: DuckLakeAttachmentManager = store._ducklake_attachment  # type: ignore[attr-defined]
    preview_conn = PreviewConnection()
    attachment.configure(preview_conn)
    return preview_conn.cursor().commands


if __name__ == "__main__":
    ducklake_store = build_store()
    print("DuckLake store initialised. Extensions:", ducklake_store.extensions)
    print("\nPreview of DuckLake ATTACH SQL:")
    for line in preview_attachment_sql(ducklake_store):
        print("  ", line)
