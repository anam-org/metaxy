"""Tests for metaxy.ext.rivers.resources."""

import pytest

import metaxy as mx
from metaxy.ext.rivers.resources import MetaxyResource


def test_metaxy_resource_store_before_setup_raises(metaxy_config: mx.MetaxyConfig):
    """Accessing .store before setup() is called should raise RuntimeError."""
    resource = MetaxyResource()

    with pytest.raises(RuntimeError, match="setup"):
        _ = resource.store


def test_metaxy_resource_setup_and_teardown(metaxy_config: mx.MetaxyConfig):
    """setup() should make .store available; teardown() should clear it."""
    resource = MetaxyResource()

    resource.setup()
    assert resource.store is not None
    assert isinstance(resource.store, mx.MetadataStore)

    resource.teardown()
    with pytest.raises(RuntimeError, match="setup"):
        _ = resource.store


def test_metaxy_resource_uses_named_store(metaxy_config: mx.MetaxyConfig):
    """A MetaxyResource with name=None should resolve to the default ("dev") store."""
    resource = MetaxyResource(name="dev")

    resource.setup()
    assert resource.store is not None
    resource.teardown()