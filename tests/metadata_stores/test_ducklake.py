"""Tests for DuckLake metadata store configuration."""

from __future__ import annotations

from metaxy.metadata_store.ducklake import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    DuckLakeMetadataStore,
    DuckLakePostgresMetadataBackend,
    DuckLakeS3StorageBackend,
    _format_attach_options,
)


class _StubCursor:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False

    def execute(self, command: str) -> None:
        # Keep the command as executed for snapshot-style assertions.
        self.commands.append(command.strip())

    def close(self) -> None:
        self.closed = True


class _StubConnection:
    def __init__(self) -> None:
        self._cursor = _StubCursor()

    def cursor(self) -> _StubCursor:
        return self._cursor


def test_ducklake_attachment_sequence() -> None:
    """DuckLakeAttachmentManager should issue expected setup statements."""
    attachment = DuckLakeAttachmentConfig(
        metadata_backend=DuckLakePostgresMetadataBackend(
            database="ducklake_meta",
            user="ducklake",
            password="secret",
        ),
        storage_backend=DuckLakeS3StorageBackend(
            endpoint_url="https://object-store",
            bucket="ducklake",
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        ),
        alias="lake",
        plugins=["ducklake"],
        attach_options={"api_version": "0.2", "override_data_path": True},
    )

    manager = DuckLakeAttachmentManager(attachment)
    conn = _StubConnection()

    manager.configure(conn)

    commands = conn.cursor().commands
    assert commands[0] == "INSTALL ducklake;"
    assert commands[1] == "LOAD ducklake;"
    assert commands[-2].startswith("ATTACH 'ducklake:secret_lake'")
    assert commands[-1] == "USE lake;"

    options_clause = _format_attach_options(attachment.attach_options)
    assert options_clause == " (API_VERSION '0.2', OVERRIDE_DATA_PATH true)"


def test_ducklake_store_extends_extensions_list() -> None:
    """DuckLake store should ensure DuckLake plugin is loaded as an extension."""
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
    )

    store = DuckLakeMetadataStore(
        metadata_backend=metadata_backend,
        storage_backend=storage_backend,
        extensions=["json"],
        attach_options={"api_version": "0.2"},
    )

    # DuckDBMetadataStore auto-adds hashfuncs, DuckLake store must add ducklake.
    assert "hashfuncs" in store.extensions
    assert "ducklake" in store.extensions
    assert "json" in store.extensions


def test_ducklake_preview_sql_uses_public_api() -> None:
    """preview_attachment_sql should expose generated SQL without private access."""
    store = DuckLakeMetadataStore(
        metadata_backend={
            "type": "postgres",
            "database": "ducklake_meta",
            "user": "ducklake",
            "password": "secret",
        },
        storage_backend={
            "type": "s3",
            "endpoint_url": "https://object-store",
            "bucket": "ducklake",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
        },
        attach_options={"override_data_path": True},
    )

    sql_commands = store.preview_attachment_sql()
    assert sql_commands[0] == "INSTALL ducklake;"
    assert sql_commands[1] == "LOAD ducklake;"
    assert sql_commands[-1] == "USE ducklake;"


def test_format_attach_options_handles_types() -> None:
    """_format_attach_options should stringify values similarly to DuckLake resource."""
    options = {
        "api_version": "0.2",
        "override_data_path": True,
        "max_retries": 3,
        "skip": None,
    }

    clause = _format_attach_options(options)
    assert clause == " (API_VERSION '0.2', MAX_RETRIES 3, OVERRIDE_DATA_PATH true)"
