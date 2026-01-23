"""Tests for ADBC metadata store base infrastructure."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from metaxy.metadata_store.adbc import ADBCMetadataStore, ADBCMetadataStoreConfig
from metaxy.models.types import FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class ConcreteADBCStore(ADBCMetadataStore):
    """Concrete implementation for testing."""

    versioning_engine_cls = PolarsVersioningEngine

    @classmethod
    def config_model(cls):
        return ADBCMetadataStoreConfig

    def _get_driver_name(self) -> str:
        return "adbc_driver_test"

    def _get_connection_options(self) -> dict[str, Any]:
        return {"uri": self.connection_string or "test://localhost"}

    def write_metadata_to_store(self, feature_key, df, **kwargs):
        pass

    def read_metadata_in_store(self, feature, *, filters=None, **kwargs):
        return None

    def _drop_feature_metadata_impl(self, feature_key):
        pass

    def _delete_metadata_impl(self, feature_key, filters, *, current_only):
        pass

    def _has_feature_impl(self, feature):
        return False


class TestADBCMetadataStoreConfig:
    """Test ADBC configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ADBCMetadataStoreConfig()

        assert config.connection_string is None
        assert config.connection_params is None
        assert config.table_prefix is None
        assert config.max_connections == 1

    def test_config_with_connection_string(self):
        """Test configuration with connection string."""
        config = ADBCMetadataStoreConfig(
            connection_string="postgresql://localhost:5432/test",
            max_connections=8,
        )

        assert config.connection_string == "postgresql://localhost:5432/test"
        assert config.max_connections == 8

    def test_config_with_connection_params(self):
        """Test configuration with connection params."""
        config = ADBCMetadataStoreConfig(
            connection_params={"host": "localhost", "port": 5432},
            table_prefix="prod_",
        )

        assert config.connection_params == {"host": "localhost", "port": 5432}
        assert config.table_prefix == "prod_"

    def test_max_connections_validation(self):
        """Test max_connections bounds validation."""
        # Valid range
        config = ADBCMetadataStoreConfig(max_connections=8)
        assert config.max_connections == 8

        # Below minimum
        with pytest.raises(ValueError):
            ADBCMetadataStoreConfig(max_connections=0)

        # Above maximum
        with pytest.raises(ValueError):
            ADBCMetadataStoreConfig(max_connections=129)


class TestADBCMetadataStore:
    """Test ADBC base class functionality."""

    def test_init_with_connection_string(self):
        """Test initialization with connection string."""
        store = ConcreteADBCStore(
            connection_string="postgresql://localhost:5432/test",
            max_connections=4,
        )

        assert store.connection_string == "postgresql://localhost:5432/test"
        assert store.max_connections == 4
        assert store.table_prefix == ""
        assert not store._is_open

    def test_init_with_connection_params(self):
        """Test initialization with connection params."""
        store = ConcreteADBCStore(
            connection_params={"host": "localhost", "port": 5432},
            table_prefix="test_",
        )

        assert store.connection_params == {"host": "localhost", "port": 5432}
        assert store.table_prefix == "test_"

    def test_get_default_hash_algorithm(self):
        """Test default hash algorithm."""
        store = ConcreteADBCStore()
        assert store._get_default_hash_algorithm() == HashAlgorithm.MD5

    def test_table_name_without_prefix(self):
        """Test table name generation without prefix."""
        store = ConcreteADBCStore()
        feature_key = FeatureKey(["test", "feature"])

        table_name = store._table_name(feature_key)
        assert table_name == "test__feature"

    def test_table_name_with_prefix(self):
        """Test table name generation with prefix."""
        store = ConcreteADBCStore(table_prefix="prod_")
        feature_key = FeatureKey(["test", "feature"])

        table_name = store._table_name(feature_key)
        assert table_name == "prod_test__feature"

    def test_display_with_connection_string(self):
        """Test display string with connection string."""
        store = ConcreteADBCStore(connection_string="postgresql://user:pass@localhost:5432/test")

        display = store.display()
        assert "ADBC" in display
        assert "adbc_driver_test" in display
        assert "****" in display  # Password should be hidden
        assert "pass" not in display  # Password should not appear

    def test_display_without_connection_string(self):
        """Test display string without connection string."""
        store = ConcreteADBCStore()

        display = store.display()
        assert display == "ADBC(adbc_driver_test)"

    def test_open_context_manager(self):
        """Test open context manager creates connection."""
        # Mock the adbc_driver_manager module
        import sys

        # Create mock module
        mock_adbc = MagicMock()
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_adbc.AdbcDatabase.return_value = mock_db
        mock_adbc.AdbcConnection.return_value = mock_conn

        # Inject into sys.modules
        sys.modules["adbc_driver_manager"] = mock_adbc

        try:
            store = ConcreteADBCStore(connection_string="test://localhost")

            assert not store._is_open
            assert store._context_depth == 0

            with store.open("read"):
                assert store._is_open
                assert store._context_depth == 1
                assert store._conn is not None
                assert store._database is not None

            # Connection should be closed after exit
            assert not store._is_open
            assert store._context_depth == 0
            mock_conn.close.assert_called_once()

        finally:
            # Clean up
            del sys.modules["adbc_driver_manager"]

    def test_nested_context_managers(self):
        """Test nested context managers reuse connection."""
        # Mock the adbc_driver_manager module
        import sys

        # Create mock module
        mock_adbc = MagicMock()
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_adbc.AdbcDatabase.return_value = mock_db
        mock_adbc.AdbcConnection.return_value = mock_conn

        # Inject into sys.modules
        sys.modules["adbc_driver_manager"] = mock_adbc

        try:
            store = ConcreteADBCStore()

            with store.open("read"):
                assert store._context_depth == 1

                # Nested open should increment depth but not create new connection
                with store.open("write"):
                    assert store._context_depth == 2
                    assert store._is_open

                # Back to depth 1, still open
                assert store._context_depth == 1
                assert store._is_open

            # Fully closed
            assert store._context_depth == 0
            assert not store._is_open

            # Connection created once, closed once
            mock_adbc.AdbcDatabase.assert_called_once()
            mock_adbc.AdbcConnection.assert_called_once()
            mock_conn.close.assert_called_once()

        finally:
            # Clean up
            del sys.modules["adbc_driver_manager"]

    def test_config_model_is_abstract(self):
        """Test that config_model must be implemented."""
        # ConcreteADBCStore implements it, should work
        assert ConcreteADBCStore.config_model() == ADBCMetadataStoreConfig

    def test_driver_name_is_abstract(self):
        """Test that _get_driver_name must be implemented."""
        # ConcreteADBCStore implements it, should work
        store = ConcreteADBCStore()
        assert store._get_driver_name() == "adbc_driver_test"

    def test_connection_options_is_abstract(self):
        """Test that _get_connection_options must be implemented."""
        # ConcreteADBCStore implements it, should work
        store = ConcreteADBCStore(connection_string="test://localhost")
        options = store._get_connection_options()
        assert "uri" in options
        assert options["uri"] == "test://localhost"


class TestADBCFromConfig:
    """Test creating ADBC stores from configuration."""

    def test_from_config_basic(self):
        """Test creating store from config."""
        config = ADBCMetadataStoreConfig(
            connection_string="postgresql://localhost/test",
            max_connections=4,
        )

        store = ConcreteADBCStore.from_config(config)

        assert store.connection_string == "postgresql://localhost/test"
        assert store.max_connections == 4

    def test_from_config_with_prefix(self):
        """Test creating store from config with table prefix."""
        config = ADBCMetadataStoreConfig(
            table_prefix="staging_",
            max_connections=2,
        )

        store = ConcreteADBCStore.from_config(config)

        assert store.table_prefix == "staging_"
        assert store.max_connections == 2
