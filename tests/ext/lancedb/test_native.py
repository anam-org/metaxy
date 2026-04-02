"""LanceDB metadata store tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

import polars as pl
import pytest

from metaxy import FeatureKey, HashAlgorithm
from metaxy.config import MetaxyConfig
from metaxy.ext.lancedb import LanceDBMetadataStore
from metaxy.metadata_store import MetadataStore, StoreNotOpenError
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_definition import FeatureDefinition
from metaxy.utils import collect_to_polars
from tests.metadata_stores.shared import (
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.mark.lancedb
@pytest.mark.polars
class TestLanceDB(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, tmp_path: Path) -> MetadataStore:
        return LanceDBMetadataStore(
            uri=tmp_path / "lancedb_store",
            hash_algorithm=HashAlgorithm.XXHASH64,
        )

    @pytest.fixture
    def named_store(self, tmp_path: Path) -> MetadataStore:
        return LanceDBMetadataStore(
            uri=tmp_path / "lancedb_store",
            hash_algorithm=HashAlgorithm.XXHASH64,
            name="dev",
        )


@pytest.fixture(autouse=True)
def _lancedb_project(config: MetaxyConfig):
    """Ensure LanceDB tests run with the same project as test features."""
    original_config = MetaxyConfig.get()
    project_config = config.model_copy(update={"project": "test_stores"})
    MetaxyConfig.set(project_config)
    yield
    MetaxyConfig.set(original_config)


@pytest.fixture
def lancedb_store(tmp_path: Path) -> LanceDBMetadataStore:
    """A LanceDBMetadataStore backed by a temporary local directory."""
    return LanceDBMetadataStore(tmp_path / "lancedb")


def test_lancedb_table_naming(
    lancedb_store: LanceDBMetadataStore, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that feature keys are converted to table names correctly."""
    feature_cls = test_features["UpstreamFeatureA"]

    with lancedb_store.open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)

        # Verify table was created (uses optimized _table_exists internally)
        assert store.has_feature(feature_cls, check_fallback=False)


def test_lancedb_conn_property_enforcement(
    lancedb_store: LanceDBMetadataStore, test_graph: FeatureGraph, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that conn property enforces store is open."""
    # Should raise when accessing conn while closed
    with pytest.raises(StoreNotOpenError, match="LanceDB connection is not open"):
        _ = lancedb_store.conn

    with lancedb_store:
        conn = lancedb_store.conn
        assert conn is not None


def test_lancedb_remote_uri_no_mkdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that open() doesn't call mkdir for remote URIs."""
    mkdir_called = []

    def mock_mkdir(*args, **kwargs):
        mkdir_called.append(True)

    original_mkdir = Path.mkdir
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    mock_lancedb = Mock()
    mock_conn = MagicMock()
    mock_lancedb.connect = Mock(return_value=mock_conn)

    remote_uris = [
        "s3://prod-bucket/metadata",
        "db://my-database",
        "https://remote-server.com/lancedb",
        "gs://gcs-bucket/data",
    ]

    for uri in remote_uris:
        mkdir_called.clear()
        store = LanceDBMetadataStore(uri)

        with monkeypatch.context() as m:
            m.setattr("lancedb.connect", mock_lancedb.connect)
            with store.open():
                assert len(mkdir_called) == 0, f"mkdir should not be called for remote URI: {uri}"
                assert store._conn is not None

    monkeypatch.setattr(Path, "mkdir", original_mkdir)


def test_lancedb_local_path_calls_mkdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that open() calls mkdir for local filesystem paths."""
    mkdir_calls = []

    original_mkdir = Path.mkdir

    def mock_mkdir(self, *args, **kwargs):
        mkdir_calls.append((str(self), args, kwargs))
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    mock_lancedb = Mock()
    mock_conn = Mock()
    mock_lancedb.connect = Mock(return_value=mock_conn)

    local_path = tmp_path / "local_lancedb"
    store = LanceDBMetadataStore(str(local_path))

    with monkeypatch.context() as m:
        m.setattr("lancedb.connect", mock_lancedb.connect)
        with store.open():
            assert len(mkdir_calls) == 1, "mkdir should be called once for local path"
            assert str(local_path) in mkdir_calls[0][0]
            assert mkdir_calls[0][2].get("parents") is True
            assert mkdir_calls[0][2].get("exist_ok") is True


def test_lancedb_connection_string_variations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the uri argument is correctly passed to lancedb.connect."""
    mock_lancedb = Mock()
    connect_calls = []

    def mock_connect(uri, **kwargs):
        connect_calls.append((uri, kwargs))
        return Mock()

    mock_lancedb.connect = mock_connect

    test_uris = [
        "s3://bucket/path",
        "db://my-db",
        str(tmp_path / "local"),
        "./relative/path",
    ]

    for uri in test_uris:
        connect_calls.clear()
        store = LanceDBMetadataStore(uri)

        with monkeypatch.context() as m:
            m.setattr("lancedb.connect", mock_connect)
            try:
                with store.open():
                    pass
            except Exception:
                pass  # Ignore errors from mock connection

        assert len(connect_calls) == 1
        assert connect_calls[0][0] == uri


def test_lancedb_sanitize_path() -> None:
    """Test that sanitize_uri properly masks credentials in URIs."""
    from metaxy.metadata_store.utils import sanitize_uri

    # URIs without credentials should pass through unchanged
    assert sanitize_uri("s3://bucket/path") == "s3://bucket/path"
    assert sanitize_uri("db://database") == "db://database"

    # Local paths should pass through unchanged
    assert sanitize_uri("./local/path") == "./local/path"
    assert sanitize_uri("/absolute/path") == "/absolute/path"

    # URIs with credentials should mask them
    assert sanitize_uri("db://user:pass@host/db") == "db://***:***@host/db"
    assert sanitize_uri("https://admin:secret@host:8000/api") == "https://***:***@host:8000/api"
    assert sanitize_uri("s3://key:secret@bucket/path") == "s3://***:***@bucket/path"

    # Username only (no password)
    assert sanitize_uri("db://user@host/db") == "db://***:@host/db"


def test_lancedb_display_masks_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that display() masks credentials in URIs."""
    # Mock lancedb.connect to avoid actual connection
    mock_lancedb = Mock()
    mock_conn = Mock()
    mock_lancedb.connect = Mock(return_value=mock_conn)

    # Test with URI containing credentials
    store = LanceDBMetadataStore("db://admin:password@localhost/mydb")

    # Check display before opening (when _is_open is False)
    display = store.display()
    assert "admin" not in display
    assert "password" not in display
    assert "***:***@localhost" in display


def test_lancedb_has_feature_without_listing_tables(
    lancedb_store: LanceDBMetadataStore, test_features: dict[str, FeatureDefinition]
) -> None:
    """Test that has_feature checks existence without listing all tables.

    This is a performance optimization - using open_table() with try/except
    is more efficient than listing all tables, especially on S3/remote storage.
    """
    feature_cls = test_features["UpstreamFeatureA"]
    nonexistent_key = FeatureKey(["nonexistent", "feature"])

    with lancedb_store.open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1],
                "metaxy_provenance_by_field": [{"frames": "h1", "audio": "h1"}],
            }
        )
        store.write(feature_cls, metadata)

        original_table_names = store.conn.table_names
        table_names_mock = Mock(side_effect=original_table_names)
        store.conn.table_names = table_names_mock

        assert store.has_feature(feature_cls, check_fallback=False)
        assert not store.has_feature(nonexistent_key, check_fallback=False)

        table_names_mock.assert_not_called()


def test_lancedb_s3_storage_options_passed(
    s3_bucket_and_storage_options: tuple[str, dict[str, str]], test_features: dict[str, FeatureDefinition]
) -> None:
    """Verify storage_options are passed to LanceDB operations with moto-backed S3.

    This ensures object store credentials are correctly forwarded to lance-rs.
    Tests the full S3 integration: connect, write, read, verify.
    """
    bucket_name, storage_options = s3_bucket_and_storage_options
    store_uri = f"s3://{bucket_name}/lancedb_store"
    feature_cls = test_features["UpstreamFeatureA"]

    with LanceDBMetadataStore(store_uri, connect_kwargs={"storage_options": storage_options}).open("w") as store:
        metadata = pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                ],
            }
        )
        store.write(feature_cls, metadata)

        assert store.has_feature(feature_cls, check_fallback=False)

        result = collect_to_polars(store.read(feature_cls))
        assert len(result) == 2
        assert set(result["sample_uid"].to_list()) == {1, 2}
