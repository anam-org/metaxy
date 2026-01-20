---
title: "Testing Features"
description: "Testing Metaxy features."
---

# Testing Metaxy Features

This guide covers patterns for testing your features when using Metaxy.

## Graph Isolation

By default, Metaxy uses a single global feature [graph][metaxy.FeatureGraph] where all features register themselves automatically.
During testing, you might want to construct your own, clean and isolated graphs.

### Using Isolated Graphs

Always use isolated graphs in tests:

```python
@pytest.fixture(autouse=True)
def graph():
    with FeatureGraph().use():
        yield graph


def test_my_feature(graph: FeatureGraph):
    class MyFeature(Feature, spec=...):
        pass

    # Test operations here

    # inspect the graph object if needed
```

The context manager ensures all feature registrations within the block use the test graph instead of the global one.
Multiple graphs can exist at the same time, but only one will be used for feature registration.

### Graph Context Management

The active graph uses context variables to support multiple graphs:

```python
# Default global graph (used in production)
graph = FeatureGraph()

# Get active graph
active = FeatureGraph.get_active()

# Use custom graph temporarily
with custom_graph.use():
    # All operations use custom_graph
    pass
```

This enables:

- **Isolated testing**: Each test gets its own feature registry
- **Migration testing**: Load historical graphs for migration scenarios
- **Multi-environment testing**: Test different feature configurations

## Testing Metadata Store Operations

### Testing with Different Backends

Use parametrized tests to verify behavior across backends:

```python
import pytest


@pytest.mark.parametrize(
    "store_cls",
    [
        DuckDBMetadataStore,
        DeltaMetadataStore,
    ],
)
def test_store_behavior(store_cls, tmp_path):
    # Use tmp_path for file-based stores
    store_kwargs = {}
    if store_cls == DuckDBMetadataStore:
        store_kwargs["database"] = tmp_path / "test.duckdb"
    elif store_cls == DeltaMetadataStore:
        store_kwargs["root_path"] = tmp_path / "delta_store"

    with store_cls(**store_kwargs) as store:
        # Test your feature operations
        pass
```

### Suppressing AUTO_CREATE_TABLES Warnings

When testing with `auto_create_tables=True`, Metaxy emits warnings to remind you not to use this in production. These warnings are important for production safety, but can clutter test output.

To suppress these warnings in your test suite, use pytest's `filterwarnings` configuration:

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

The warning is still emitted (important for production awareness), but pytest filters it from test output.

**Testing the Warning Itself**

If you need to verify that the warning is actually emitted, use `pytest.warns()`:

```python
import pytest


def test_auto_create_tables_warning():
    with pytest.warns(
        UserWarning, match=r"AUTO_CREATE_TABLES is enabled.*do not use in production"
    ):
        with DuckDBMetadataStore(":memory:", auto_create_tables=True) as store:
            pass  # Warning is emitted and captured
```

This works even with `filterwarnings` configured, because `pytest.warns()` explicitly captures and verifies the warning.

### Programmatic Metaxy Configuration

When configuring Metaxy programmatically in Python code, use the following pattern (we also showcase plugin configuration):

```python
from metaxy.config import MetaxyConfig
from metaxy.ext.sqlmodel import SQLModelPluginConfig

with MetaxyConfig(
    ext={
        "sqlmodel": SQLModelPluginConfig(
            enable=True,
            infer_db_table_names=True,
        )
    }
).use() as config:
    sqlmodel_config = MetaxyConfig.get_plugin("sqlmodel", SQLModelPluginConfig)
    # your code here
```

When testing Metaxy code, it's best if this setup is performed via `pytest` fixtures.

The plugin configuration is accessed via a dictionary where keys are plugin names and values are plugin-specific configuration objects.
