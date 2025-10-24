"""Tests for DuckLake integration via DuckDB metadata store."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from metaxy._utils import collect_to_polars
from metaxy.metadata_store._ducklake_support import (
    DuckLakeAttachmentConfig,
    DuckLakeAttachmentManager,
    _PreviewConnection,
    format_attach_options,
)
from metaxy.metadata_store.duckdb import DuckDBMetadataStore


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

    options_clause = format_attach_options(attachment.attach_options)
    assert options_clause == " (API_VERSION '0.2', OVERRIDE_DATA_PATH true)"


def test_format_attach_options_handles_types() -> None:
    """_format_attach_options should stringify values similarly to DuckLake resource."""
    options = {
        "api_version": "0.2",
        "override_data_path": True,
        "max_retries": 3,
        "skip": None,
    }

    clause = format_attach_options(options)
    assert clause == " (API_VERSION '0.2', MAX_RETRIES 3, OVERRIDE_DATA_PATH true)"


def _ducklake_config_payload() -> dict[str, object]:
    return {
        "metadata_backend": {
            "type": "postgres",
            "database": "ducklake_meta",
            "user": "ducklake",
            "password": "secret",
        },
        "storage_backend": {
            "type": "s3",
            "endpoint_url": "https://object-store",
            "bucket": "ducklake",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
        },
        "attach_options": {"override_data_path": True},
    }


def test_duckdb_store_accepts_ducklake_config() -> None:
    """DuckDBMetadataStore should accept ducklake configuration inline."""
    store = DuckDBMetadataStore(
        database=":memory:",
        extensions=["json"],
        ducklake=_ducklake_config_payload(),
    )

    assert "ducklake" in store.extensions
    commands = store.preview_ducklake_sql()
    assert commands[0] == "INSTALL ducklake;"
    assert commands[1] == "LOAD ducklake;"
    assert commands[-1] == "USE ducklake;"


def test_duckdb_store_preview_via_config_manager() -> None:
    """DuckDBMetadataStore exposes attachment manager helpers when configured."""
    store = DuckDBMetadataStore(
        database=":memory:",
        ducklake=_ducklake_config_payload(),
    )

    manager = store.ducklake_attachment
    preview = manager.preview_sql()
    assert preview[0] == "INSTALL ducklake;"
    assert preview[-1] == "USE ducklake;"


@pytest.mark.usefixtures("test_graph")
def test_ducklake_store_read_write_roundtrip(
    tmp_path, test_features, monkeypatch
) -> None:
    """DuckLake-configured store should still support read/write API."""
    recorded_commands: list[str] = []
    original_configure = DuckLakeAttachmentManager.configure

    def fake_configure(self, conn):
        preview_conn = _PreviewConnection()
        original_configure(self, preview_conn)
        recorded_commands.extend(preview_conn.cursor().commands)

    monkeypatch.setattr(DuckLakeAttachmentManager, "configure", fake_configure)

    db_path = tmp_path / "ducklake_roundtrip.duckdb"
    metadata_path = tmp_path / "ducklake_catalog.duckdb"
    storage_dir = tmp_path / "ducklake_storage"

    ducklake_config = {
        "alias": "lake",
        "metadata_backend": {"type": "duckdb", "path": str(metadata_path)},
        "storage_backend": {"type": "local", "path": str(storage_dir)},
    }

    feature = test_features["UpstreamFeatureA"]
    payload = pl.DataFrame(
        {
            "sample_id": [1, 2],
            "data_version": [
                {"frames": "hash_1", "audio": "hash_1"},
                {"frames": "hash_2", "audio": "hash_2"},
            ],
        }
    )

    store = DuckDBMetadataStore(
        database=db_path,
        extensions=["json"],
        ducklake=ducklake_config,
    )

    with store:
        store.write_metadata(feature, payload)
        result = collect_to_polars(store.read_metadata(feature))

    actual = result.sort("sample_id").select(["sample_id", "data_version"])
    expected = payload.sort("sample_id")
    assert_frame_equal(actual, expected)

    assert recorded_commands[:2] == ["INSTALL ducklake;", "LOAD ducklake;"]
    assert any(cmd.startswith("ATTACH 'ducklake:") for cmd in recorded_commands)
    assert recorded_commands[-1] == "USE lake;"