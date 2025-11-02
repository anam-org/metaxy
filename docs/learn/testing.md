# Testing Metaxy Features

This guide covers patterns for testing your features when using Metaxy.

## Graph Isolation

By default, Metaxy uses a single global feature graph where all features register themselves automatically. In testing, you need isolated graphs to prevent test interference.

### Using Isolated Graphs

Always use isolated graphs in tests:

```python
def test_my_feature():
    test_graph = FeatureGraph()
    with test_graph.use():

        class MyFeature(Feature, spec=...):
            pass

        # Test operations here
```

The context manager ensures all feature registrations within the block use the test graph instead of the global one.

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

### Context Manager Pattern

Stores must be used as context managers to ensure proper resource cleanup:

```python
def test_metadata_operations():
    with InMemoryMetadataStore() as store:
        # Create test data
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": {...},
                "feature_version": "abc123",
            }
        )

        # Write metadata
        store.write_metadata(MyFeature, df)

        # Read and verify
        result = store.read_metadata(MyFeature)
        assert len(result) == 3
```

### Testing with Different Backends

Use parametrized tests to verify behavior across backends:

```python
import pytest


@pytest.mark.parametrize(
    "store_cls",
    [
        InMemoryMetadataStore,
        DuckDBMetadataStore,
    ],
)
def test_store_behavior(store_cls, tmp_path):
    # Use tmp_path for file-based stores
    store_kwargs = {}
    if store_cls != InMemoryMetadataStore:
        store_kwargs["path"] = tmp_path / "test.db"

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

## Testing Custom Alignment

If your feature overrides `load_input()` for custom alignment, test it thoroughly:

```python
def test_custom_alignment():
    # Prepare test data
    current = pl.DataFrame({"sample_uid": [1, 2, 3], "custom_field": ["a", "b", "c"]})

    upstream = {
        "video_feature": pl.DataFrame(
            {"sample_uid": [2, 3, 4], "metaxy_provenance_by_field": {...}}
        )
    }

    # Test alignment logic
    result = MyFeature.load_input(current, upstream)

    # Verify behavior
    assert set(result["sample_uid"].to_list()) == {2, 3}  # Inner join
    assert "custom_field" in result.columns  # Custom fields preserved
```

## Testing Feature Dependencies

Verify that dependencies are correctly defined:

```python
def test_feature_dependencies():
    test_graph = FeatureGraph()

    with test_graph.use():
        # Define upstream feature
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
            ),
        ):
            pass

        # Define downstream feature with dependency
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["processed"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature=FeatureKey(["upstream"]),
                                fields=[FieldKey(["data"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

        # Verify graph structure
        assert len(test_graph.features_by_key) == 2
        assert UpstreamFeature in test_graph.get_downstream(DownstreamFeature)
```

## Testing Migrations

### Simulating Feature Changes

Test how your features behave when definitions change:

```python
def test_migration_scenario():
    # Initial version
    graph_v1 = FeatureGraph()
    with graph_v1.use():

        class MyFeatureV1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["my_feature"]),
                fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
            ),
        ):
            pass

    # Record initial state
    with InMemoryMetadataStore() as store:
        store.record_feature_graph_snapshot(graph_v1)

        # Modified version
        graph_v2 = FeatureGraph()
        with graph_v2.use():

            class MyFeatureV2(
                Feature,
                spec=FeatureSpec(
                    key=FeatureKey(["my_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="2")],
                ),
            ):
                pass

        # Verify version change detected
        with graph_v2.use():
            changes = store.detect_feature_changes(MyFeatureV2)
            assert changes is not None
```

### Testing Migration Idempotency

Ensure migrations can be safely re-run:

```python
def test_migration_idempotency():
    with InMemoryMetadataStore() as store:
        # Apply migration twice
        apply_migration(store, migration)
        result1 = store.read_metadata(MyFeature)

        apply_migration(store, migration)  # Should be no-op
        result2 = store.read_metadata(MyFeature)

        # Results should be identical
        assert result1.equals(result2)
```

## Best Practices

1. **Always use isolated graphs** - Never rely on the global graph in tests
2. **Use context managers** - Ensure proper cleanup of stores and resources
3. **Test across backends** - Verify features work with different metadata stores
4. **Test edge cases** - Empty data, missing dependencies, version conflicts
5. **Mock external dependencies** - Isolate tests from external services
6. **Verify determinism** - Feature versions should be consistent across runs
