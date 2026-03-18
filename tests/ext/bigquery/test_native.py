"""BigQuery metadata store tests."""

from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest
from metaxy_testing.models import SampleFeature

from metaxy.ext.bigquery import BigQueryMetadataStore
from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import FeatureGraph
from metaxy.versioning.types import HashAlgorithm
from tests.metadata_stores.shared import (
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    IbisMapTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
)


@pytest.mark.ibis
@pytest.mark.native
@pytest.mark.bigquery
class TestBigQuery(
    CRUDTests,
    DeletionTests,
    DisplayTests,
    FilterTests,
    IbisMapTests,
    MapDtypeTests,
    ResolveUpdateTests,
    VersioningTests,
    WriteTests,
):
    @pytest.fixture
    def store(self, request: pytest.FixtureRequest) -> MetadataStore:
        return BigQueryMetadataStore(
            project_id=request.getfixturevalue("bigquery_project_id"),
            dataset_id=request.getfixturevalue("bigquery_dataset"),
            hash_algorithm=HashAlgorithm.MD5,
            auto_create_tables=True,
        )

    @pytest.fixture
    def named_store(self, request: pytest.FixtureRequest) -> MetadataStore:
        return BigQueryMetadataStore(
            project_id=request.getfixturevalue("bigquery_project_id"),
            dataset_id=request.getfixturevalue("bigquery_dataset"),
            hash_algorithm=HashAlgorithm.MD5,
            auto_create_tables=True,
            name="bigquery-test",
        )


# ---------------------------------------------------------------------------
# BigQuery-specific unit tests (no live connection required)
# ---------------------------------------------------------------------------


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
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
    )

    assert store.project_id == "test-project"
    assert store.dataset_id == "test_dataset"


def test_bigquery_initialization_with_credentials_path():
    with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds:
        mock_creds.return_value = Mock()

        BigQueryMetadataStore(
            project_id="test-project",
            dataset_id="test_dataset",
            credentials_path="/path/to/creds.json",
        )

        mock_creds.assert_called_once_with(
            "/path/to/creds.json",
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )


def test_bigquery_initialization_with_invalid_credentials_path():
    with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds:
        mock_creds.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError, match="Service account credentials file not found"):
            BigQueryMetadataStore(
                project_id="test-project",
                dataset_id="test_dataset",
                credentials_path="/nonexistent/creds.json",
            )

        mock_creds.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ValueError, match="Invalid service account credentials file"):
            BigQueryMetadataStore(
                project_id="test-project",
                dataset_id="test_dataset",
                credentials_path="/invalid/creds.json",
            )


def test_bigquery_project_id_from_connection_params():
    store = BigQueryMetadataStore(
        dataset_id="test_dataset",
        connection_params={
            "project_id": "test-project",
            "location": "US",
        },
    )

    assert store.project_id == "test-project"
    assert store.dataset_id == "test_dataset"
    assert store.connection_params["location"] == "US"


def test_bigquery_explicit_project_id_overrides_connection_params():
    store = BigQueryMetadataStore(
        project_id="explicit-project",
        dataset_id="test_dataset",
        connection_params={
            "project_id": "params-project",
            "location": "EU",
        },
    )

    assert store.project_id == "explicit-project"
    assert store.connection_params["location"] == "EU"


def test_bigquery_initialization_missing_project():
    with pytest.raises(ValueError, match="Must provide project_id"):
        BigQueryMetadataStore(dataset_id="test_dataset")


def test_bigquery_initialization_missing_dataset():
    with pytest.raises(ValueError, match="dataset_id is required"):
        BigQueryMetadataStore(project_id="test-project")


def test_bigquery_default_hash_algorithm():
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
    )

    assert store.hash_algorithm == HashAlgorithm.MD5


def test_bigquery_display_string():
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
    )

    display = store.display()
    assert "BigQueryMetadataStore" in display
    assert "test-project" in display
    assert "test_dataset" in display


def test_bigquery_sqlalchemy_url():
    store = BigQueryMetadataStore(
        project_id="my-project",
        dataset_id="my_dataset",
    )

    assert store.sqlalchemy_url == "bigquery://my-project/my_dataset"


def test_bigquery_location_parameter():
    store = BigQueryMetadataStore(
        project_id="test-project",
        dataset_id="test_dataset",
        location="EU",
    )

    assert store.connection_params.get("location") == "EU"


def test_bigquery_config_instantiation():
    from metaxy.config import MetaxyConfig, StoreConfig

    store = MetaxyConfig(
        stores={
            "bigquery_store": StoreConfig(
                type="metaxy.ext.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "test-project",
                    "dataset_id": "test_dataset",
                },
            )
        }
    ).get_store("bigquery_store")

    assert isinstance(store, BigQueryMetadataStore)
    assert store.project_id == "test-project"
    assert store.dataset_id == "test_dataset"


def test_bigquery_config_with_hash_algorithm():
    from metaxy.config import MetaxyConfig, StoreConfig

    # Test default from config system is XXHASH64 (not FARMHASH)
    # When no hash_algorithm is specified in config, the config system defaults to XXHASH64
    config_default = MetaxyConfig(
        stores={
            "bigquery_store": StoreConfig(
                type="metaxy.ext.bigquery.BigQueryMetadataStore",
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
                type="metaxy.ext.bigquery.BigQueryMetadataStore",
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
                type="metaxy.ext.bigquery.BigQueryMetadataStore",
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
    from metaxy.config import MetaxyConfig, StoreConfig

    dev_store = MetaxyConfig(
        stores={
            "dev": StoreConfig(
                type="metaxy.ext.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "dev-project",
                    "dataset_id": "dev_dataset",
                    "fallback_stores": ["prod"],
                },
            ),
            "prod": StoreConfig(
                type="metaxy.ext.bigquery.BigQueryMetadataStore",
                config={
                    "project_id": "prod-project",
                    "dataset_id": "prod_dataset",
                },
            ),
        }
    ).get_store("dev")

    assert isinstance(dev_store, BigQueryMetadataStore)
    assert len(dev_store.fallback_stores) == 1
    assert isinstance(dev_store.fallback_stores[0], BigQueryMetadataStore)


@pytest.mark.integration
def test_bigquery_table_operations(
    mock_bigquery_connection: MagicMock, test_graph: FeatureGraph, test_features: dict[str, type[SampleFeature]]
):
    with patch("ibis.bigquery.connect", return_value=mock_bigquery_connection):
        with BigQueryMetadataStore(
            project_id="test-project",
            dataset_id="test_dataset",
        ).open("w") as store:
            store._write_feature = MagicMock()  # ty: ignore[invalid-assignment]

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
            store.write(test_features["UpstreamFeatureA"], metadata)

            assert store._write_feature.called  # ty: ignore[unresolved-attribute]
            call_args = store._write_feature.call_args[0]  # ty: ignore[unresolved-attribute]
            assert call_args[0].table_name == "test_stores__upstream_a"
