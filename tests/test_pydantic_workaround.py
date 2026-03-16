import os
import sys
import typing
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field
from pydantic_settings import BaseSettings

from metaxy.config import MetaxyConfig, SafeEnvSettingsSource

class MockField:
    """A mock field that lacks the 'annotation' attribute to simulate Python 3.10 typing.Any."""
    pass

def test_safe_env_settings_source_catches_attribute_error():
    """Verify that SafeEnvSettingsSource correctly catches AttributeError when accessing .annotation."""
    class DummySettings(BaseSettings):
        pass

    source = SafeEnvSettingsSource(DummySettings)
    mock_field = MockField()
    
    # Verify that accessing .annotation on MockField raises AttributeError
    with pytest.raises(AttributeError):
        _ = getattr(mock_field, "annotation")

    # SafeEnvSettingsSource.field_is_complex should catch the error and return True
    result = source.field_is_complex(mock_field)
    print(f"field_is_complex result: {result}")
    assert result is True
    
    # SafeEnvSettingsSource._field_is_complex should catch the error
    result_tuple = source._field_is_complex(mock_field)
    print(f"_field_is_complex result: {result_tuple}")
    # The first element should be True (is complex)
    assert result_tuple[0] is True
    
    # SafeEnvSettingsSource.decode_complex_value should catch the error and return the value as-is
    assert source.decode_complex_value("test_field", mock_field, "test_value") == "test_value"

def test_safe_env_settings_source_normal_behavior():
    """Verify that SafeEnvSettingsSource behaves normally when no AttributeError occurs."""
    class DummySettings(BaseSettings):
        test_val: int = 1

    source = SafeEnvSettingsSource(DummySettings)
    
    # Use a real FieldInfo-like object that has the attribute
    real_field = MagicMock()
    real_field.annotation = int
    
    # Should delegate to super() which normally returns False for int
    assert source.field_is_complex(real_field) is False

def test_metaxy_config_uses_safe_source(monkeypatch):
    """Verify that MetaxyConfig.settings_customise_sources returns our SafeEnvSettingsSource."""
    class MockSettings(BaseSettings):
        pass

    mock_init = MagicMock()
    mock_env = MagicMock()
    mock_env.case_sensitive = True
    mock_env.env_prefix = "METAXY_"
    mock_env.env_nested_delimiter = "__"
    # Ensure we don't pass attributes that the actual EnvSettingsSource doesn't support
    # (Removing env_parse_none if it was somehow present on the mock)
    if hasattr(mock_env, "env_parse_none"):
        delattr(mock_env, "env_parse_none")
        
    mock_dotenv = MagicMock()
    mock_file = MagicMock()
    
    # Correct call: don't pass MetaxyConfig as a positional argument twice
    sources = MetaxyConfig.settings_customise_sources(
        MockSettings,
        mock_init,
        mock_env,
        mock_dotenv,
        mock_file
    )
    
    # sources should be (init, safe_env, toml)
    assert len(sources) == 3
    assert isinstance(sources[1], SafeEnvSettingsSource)
    assert sources[1].case_sensitive is True
    assert sources[1].env_prefix == "METAXY_"

def test_integration_with_nested_env_vars(monkeypatch):
    """Integration test to ensure nested env vars are parsed correctly via SafeEnvSettingsSource."""
    monkeypatch.setenv("METAXY_STORES__TEST__TYPE", "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore")
    monkeypatch.setenv("METAXY_STORES__TEST__CONFIG__DATABASE", ":memory:")
    
    config = MetaxyConfig.load(search_parents=False)
    
    assert "test" in config.stores
    assert config.stores["test"].type == "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"
    assert config.stores["test"].config["database"] == ":memory:"
