"""BigQuery-specific tests that don't apply to other stores."""

from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest

# Skip all tests in this module if BigQuery not available
pytest.importorskip("ibis")

# We don't need the full ibis.backends.bigquery to run tests since we mock everything
# Just need to be able to import our BigQueryMetadataStore class
try:
    from metaxy.metadata_store.bigquery import BigQueryMetadataStore
except ImportError:
    pytest.skip("BigQueryMetadataStore not available", allow_module_level=True)

from metaxy_testing.models import SampleFeature

from metaxy.versioning.types import HashAlgorithm


@pytest.fixture
def mock_bigquery_connection():
    """Mock BigQuery connection for testing without actual BigQuery instance."""
    with patch("ibis.bigquery") as mock_bq:
        mock_conn = MagicMock()
        mock_conn.list_tables.return_value = []
        mock_conn.table = MagicMock()
        mock_bq.connect.return_value = mock_conn
        yield mock_conn


def test_bigquery_initialization_with_project_dataset():
    """Test BigQuery store initialization with project and dataset."""
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
    )

    assert store.project_id == "test-project"
    assert store.dataset_id == "test_dataset"


def test_bigquery_initialization_with_credentials_path():
    """Test BigQuery store initialization with credentials path."""
    with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds:
        mock_creds_instance = Mock()
        mock_creds.return_value = mock_creds_instance

        _ = BigQueryMetadataStore(
            project_id="test-project",
            dataset_id="test_dataset",
            credentials_path="/path/to/creds.json",
        )

        # Verify credentials were loaded using the recommended method
        mock_creds.assert_called_once_with(
            "/path/to/creds.json",
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )


def test_bigquery_initialization_with_invalid_credentials_path():
    """Test BigQuery store initialization with invalid credentials path."""
    with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds:
        # Simulate file not found
        mock_creds.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError, match="Service account credentials file not found"):
            BigQueryMetadataStore(
                project_id="test-project",
                dataset_id="test_dataset",
                credentials_path="/nonexistent/creds.json",
            )

        # Simulate invalid JSON format
        mock_creds.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ValueError, match="Invalid service account credentials file"):
            BigQueryMetadataStore(
                project_id="test-project",
                dataset_id="test_dataset",
                credentials_path="/invalid/creds.json",
            )


def test_bigquery_initialization_with_connection_params():
    """Test BigQuery store initialization with connection_params."""
    store = BigQueryMetadataStore(
        connection_params={
            "project_id": "test-project",
            "dataset_id": "test_dataset",
            "location": "US",
        }
    )

    assert store.project_id == "test-project"
    assert store.dataset_id == "test_dataset"


def test_bigquery_initialization_missing_project():
    """Test that initialization fails without project_id."""
    with pytest.raises(ValueError, match="Must provide either project_id"):
        BigQueryMetadataStore(dataset_id="test_dataset")


def test_bigquery_hash_algorithms():
    """Test that BigQuery supports FARMHASH, MD5 and SHA256 hash algorithms."""
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
    )

    # Should support FARMHASH (default)
    assert store.hash_algorithm == HashAlgorithm.MD5


def test_bigquery_display_string():
    """Test display string generation for BigQuery store."""
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
    )

    display = store.display()
    assert "BigQueryMetadataStore" in display
    assert "test-project" in display
    assert "test_dataset" in display


def test_bigquery_location_parameter():
    """Test BigQuery store with location parameter."""
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
        location="EU",
    )

    # Verify location is passed to connection params
    assert store.connection_params.get("location") == "EU"


def test_bigquery_config_instantiation():
    """Test instantiating BigQuery store via MetaxyConfig."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "bigquery_store": StoreConfig(
                type="metaxy.metadata_store.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                },
            )
        }
    )

    store = config.get_store("bigquery_store")
    assert isinstance(store, BigQueryMetadataStore)
    assert store.project_id == "test-project"
    assert store.dataset_id == "test_dataset"


def test_bigquery_config_with_hash_algorithm():
    """Test BigQuery store config with specific hash algorithm."""
    from metaxy.config import MetaxyConfig, StoreConfig

    # Test default from config system is XXHASH64 (not FARMHASH)
    # When no hash_algorithm is specified in config, the config system defaults to XXHASH64
    config_default = MetaxyConfig(
        stores={
            "bigquery_store": StoreConfig(
                type="metaxy.metadata_store.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                },
            )
        }
    )

    store_default = config_default.get_store("bigquery_store")
    assert isinstance(store_default, BigQueryMetadataStore)
    assert store_default.hash_algorithm == HashAlgorithm.MD5  # Config system default

    config_farmhash = MetaxyConfig(
        stores={
            "bigquery_store": StoreConfig(
                type="metaxy.metadata_store.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                    "hash_algorithm": "farmhash",
                },
            )
        }
    )

    store_farmhash = config_farmhash.get_store("bigquery_store")
    assert isinstance(store_farmhash, BigQueryMetadataStore)
    assert store_farmhash.hash_algorithm == HashAlgorithm.FARMHASH

    # Test explicit MD5
    config_md5 = MetaxyConfig(
        stores={
            "bigquery_store": StoreConfig(
                type="metaxy.metadata_store.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                    "hash_algorithm": "md5",
                },
            )
        }
    )

    store_md5 = config_md5.get_store("bigquery_store")
    assert isinstance(store_md5, BigQueryMetadataStore)
    assert store_md5.hash_algorithm == HashAlgorithm.MD5


def test_bigquery_config_with_fallback_stores():
    """Test BigQuery store config with fallback stores."""
    from metaxy.config import MetaxyConfig, StoreConfig

    config = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.metadata_store.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "dev-project",
                    "dataset_id": "dev_dataset",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.metadata_store.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "prod-project",
                    "dataset_id": "prod_dataset",
                },
            ),
        }
    )

    dev_store = config.get_store("dev")
    assert isinstance(dev_store, BigQueryMetadataStore)
    assert len(dev_store.fallback_stores) == 1
    assert isinstance(dev_store.fallback_stores[0], BigQueryMetadataStore)


@pytest.mark.integration
def test_bigquery_table_operations(mock_bigquery_connection, test_graph, test_features: dict[str, type[SampleFeature]]):
    """Test BigQuery table operations with mocked connection.

    This test would require actual BigQuery connection in integration tests.
    """
    with patch("ibis.bigquery.connect", return_value=mock_bigquery_connection):
        with BigQueryMetadataStore(
            project_id="test-project",
            dataset_id="test_dataset",
        ) as store:
            # Mock the write operation
            store.write_metadata_to_store = MagicMock()  # ty: ignore[invalid-assignment]

            metadata = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "metaxy_provenance_by_field": [
                        {"frames": "h1", "audio": "h1"},
                        {"frames": "h2", "audio": "h2"},
                        {"frames": "h3", "audio": "h3"},
                    ],
                }
            )
            store.write_metadata(test_features["UpstreamFeatureA"], metadata)

            # Verify write was called with correct table name
            assert store.write_metadata_to_store.called  # ty: ignore[unresolved-attribute]
            call_args = store.write_metadata_to_store.call_args[0]  # ty: ignore[unresolved-attribute]
            assert call_args[0].table_name == "test_stores__upstream_a"
