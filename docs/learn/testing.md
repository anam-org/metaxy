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

### Suppressing `AUTO_CREATE_TABLES` Warnings

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
