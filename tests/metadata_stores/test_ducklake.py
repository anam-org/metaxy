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


@pytest.mark.usefixtures("test_graph")
def test_ducklake_e2e_with_dependencies(tmp_path, test_features, monkeypatch) -> None:
    """End-to-end test for DuckLake with DuckDB catalog, local filesystem storage, and feature dependencies.

    This test exercises the full workflow:
    1. Write metadata for upstream features
    2. Write metadata for downstream features with dependencies
    3. Read metadata back and verify data versions
    4. Test metadata updates and versioning
    """
    recorded_commands: list[str] = []
    original_configure = DuckLakeAttachmentManager.configure

    def fake_configure(self, conn):
        preview_conn = _PreviewConnection()
        original_configure(self, preview_conn)
        recorded_commands.extend(preview_conn.cursor().commands)

    monkeypatch.setattr(DuckLakeAttachmentManager, "configure", fake_configure)

    # Setup paths
    db_path = tmp_path / "ducklake_e2e.duckdb"
    metadata_path = tmp_path / "ducklake_catalog.duckdb"
    storage_dir = tmp_path / "ducklake_storage"

    ducklake_config = {
        "alias": "e2e_lake",
        "metadata_backend": {"type": "duckdb", "path": str(metadata_path)},
        "storage_backend": {"type": "local", "path": str(storage_dir)},
        "attach_options": {"override_data_path": True},
    }

    # Create store
    store = DuckDBMetadataStore(
        database=db_path,
        extensions=["json"],
        ducklake=ducklake_config,
    )

    # Test 1: Write upstream feature metadata
    upstream_a = test_features["UpstreamFeatureA"]
    upstream_b = test_features["UpstreamFeatureB"]
    downstream = test_features["DownstreamFeature"]

    upstream_a_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [
                {"frames": "hash_a1", "audio": "hash_a1"},
                {"frames": "hash_a2", "audio": "hash_a2"},
                {"frames": "hash_a3", "audio": "hash_a3"},
            ],
        }
    )

    upstream_b_data = pl.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "data_version": [
                {"default": "hash_b1"},
                {"default": "hash_b2"},
                {"default": "hash_b3"},
            ],
        }
    )

    with store:
        # Write upstream features
        store.write_metadata(upstream_a, upstream_a_data)
        store.write_metadata(upstream_b, upstream_b_data)

        # Verify upstream features can be read back
        result_a = collect_to_polars(store.read_metadata(upstream_a))
        result_b = collect_to_polars(store.read_metadata(upstream_b))

        assert_frame_equal(
            result_a.sort("sample_id").select(["sample_id", "data_version"]),
            upstream_a_data.sort("sample_id"),
        )
        assert_frame_equal(
            result_b.sort("sample_id").select(["sample_id", "data_version"]),
            upstream_b_data.sort("sample_id"),
        )

        # Test 2: Write downstream feature with dependencies
        downstream_data = pl.DataFrame(
            {
                "sample_id": [1, 2, 3],
                "data_version": [
                    {"default": "hash_d1"},
                    {"default": "hash_d2"},
                    {"default": "hash_d3"},
                ],
            }
        )

        store.write_metadata(downstream, downstream_data)

        # Verify downstream feature can be read back
        result_d = collect_to_polars(store.read_metadata(downstream))
        assert_frame_equal(
            result_d.sort("sample_id").select(["sample_id", "data_version"]),
            downstream_data.sort("sample_id"),
        )

        # Test 3: List features
        features_list = store.list_features()
        assert len(features_list) == 3
        feature_keys = set(features_list)
        assert upstream_a.spec.key in feature_keys
        assert upstream_b.spec.key in feature_keys
        assert downstream.spec.key in feature_keys

        # Test 4: Update metadata (append-only write)
        # Metaxy uses immutable, append-only metadata storage
        updated_upstream_a = pl.DataFrame(
            {
                "sample_id": [4],  # Add just a new sample
                "data_version": [
                    {"frames": "hash_a4", "audio": "hash_a4"},
                ],
            }
        )

        store.write_metadata(upstream_a, updated_upstream_a)

        # Verify updated metadata - should have 4 samples total (3 + 1)
        result_updated = collect_to_polars(store.read_metadata(upstream_a))
        assert len(result_updated) == 4
        assert set(result_updated["sample_id"].to_list()) == {1, 2, 3, 4}

    # Verify DuckLake commands were executed
    assert recorded_commands[:2] == ["INSTALL ducklake;", "LOAD ducklake;"]
    assert any(cmd.startswith("ATTACH 'ducklake:") for cmd in recorded_commands)
    assert recorded_commands[-1] == "USE e2e_lake;"

    # Test 5: Verify persistence by reopening the store
    # Note: We use the same monkeypatch for consistency
    recorded_commands2: list[str] = []

    def fake_configure2(self, conn):
        preview_conn = _PreviewConnection()
        original_configure(self, preview_conn)
        recorded_commands2.extend(preview_conn.cursor().commands)

    monkeypatch.setattr(DuckLakeAttachmentManager, "configure", fake_configure2)

    store2 = DuckDBMetadataStore(
        database=db_path,
        extensions=["json"],
        ducklake=ducklake_config,
    )

    with store2:
        # Verify we can still read all features after reopening
        result_a2 = collect_to_polars(store2.read_metadata(upstream_a))
        assert len(result_a2) == 4

        result_b2 = collect_to_polars(store2.read_metadata(upstream_b))
        assert len(result_b2) == 3

        result_d2 = collect_to_polars(store2.read_metadata(downstream))
        assert len(result_d2) == 3

        # Verify feature list persists
        features_list2 = store2.list_features()
        assert len(features_list2) == 3

    # Verify DuckLake was attached again on second open
    assert len(recorded_commands2) > 0
    assert "INSTALL ducklake;" in recorded_commands2
