---
title: "Testing Metaxy Code"
description: "Testing Metaxy code: examples and best practices."
---

# Testing Metaxy Code

This guide covers patterns for testing your Metaxy code. As always, Metaxy must be explicitly initialized with [`init_metaxy`][metaxy.init_metaxy]:

<!-- skip next -->
```py
import metaxy as mx

mx.init_metaxy()
```

This is typically done in a `pytest` fixture.

## Ephemeral Configuration

The current Metaxy [configuration](/reference/configuration.md) is available via a [`MetaxyConfig.get()`][metaxy.MetaxyConfig.get] singleton.
It is often desired to provide custom Metaxy configuration in tests.

This can be achieved by constructing a `MetaxyConfig` instance and activating it via a context manager. It's best if this setup is performed via `pytest` fixtures:

```python
import pytest
import metaxy as mx


@pytest.fixture(autouse=True)
def metaxy_config():
    with mx.MetaxyConfig(project="my-project").use() as config:
        yield config


def test_my_config():
    assert mx.MetaxyConfig.get().project == "my-project"
```

The config object can be explicitly passed to `init_metaxy`. This can be used to adjust how feature discovery is performed.

### Configuring Metaxy Plugins

Plugins can be configured via a dictionary where keys are plugin names and values are plugin-specific configuration objects:

```python
from metaxy.config import MetaxyConfig
from metaxy.ext.sqlmodel import SQLModelPluginConfig

with MetaxyConfig(
    ext={
        "sqlmodel": SQLModelPluginConfig(
            enable=True,
            inject_primary_key=True,
        )
    }
).use() as cfg:
    sqlmodel_config = MetaxyConfig.get_plugin("sqlmodel", SQLModelPluginConfig)
    assert sqlmodel_config.inject_primary_key is True
```

### Multi-Project Testing

When working with multi-project setups, it's a good idea to [provide an explicit path](/reference/configuration.md/#metaxy_lock_path) to the `metaxy.lock` file.

## Ephemeral Feature Graphs

By default, Metaxy uses a single global feature [graph][metaxy.FeatureGraph] where all features are registered.
During testing, you might want to construct your own, clean and isolated feature graphs.

### Using Isolated Graphs

Always use isolated graphs in tests:

```python
import pytest
import metaxy as mx
from metaxy.models.feature import FeatureGraph


@pytest.fixture(autouse=True)
def isolated_graph():
    with FeatureGraph().use() as g:
        yield g


def test_my_feature():
    class TestFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="test/feature", id_columns=["id"])):
        id: str

    # Test operations here
    assert isolated_graph.get_feature_definition("test/feature") is not None
```

The context manager ensures all feature registrations within the block use the test graph instead of the global one.
Multiple graphs instances can be created at the same time, but only one will be used for feature registration.

## Testing with production data

It's often a good idea to setup "integration" test for data by using real data samples from production.
It's often unavoidable in data applications, as this production data may be nearly impossible to replicate or mock.

To achieve this with Metaxy, configure [fallback stores](../metadata-stores.md#fallback-stores) for your testing metadata store to pull upstream data from production.

## Suppressing `AUTO_CREATE_TABLES` Warnings

When using certain database-based (1) metadata stores with `auto_create_tables` set to `True`, Metaxy emits warnings to remind you not to use this in production.
It may be desired to suppress these warnings in your test suite.
{ .annotate }

1. pun intended

To suppress these warnings in your test suite, use `pytest`'s `filterwarnings` configuration:

```toml
# pyproject.toml
[tool.pytest.ini_options]
env = [
  "METAXY_AUTO_CREATE_TABLES=1", # Enable auto-creation in tests
]
filterwarnings = [
  "ignore:AUTO_CREATE_TABLES is enabled:UserWarning", # Suppress the warning
]
```

The warning is still emitted (important for production awareness), but `pytest` filters it from test output.
