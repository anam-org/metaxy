---
name: metaxy
description: This skill should be used when the user asks to "define a feature", "create a BaseFeature class", "track feature versions", "set up metadata store", "field-level lineage", "FieldSpec", "FeatureDep", "run metaxy CLI", "metaxy migrations", "metaxy lock", "lock features", "external features", "multi-environment", "monorepo features", "enable Map datatype", "enable_map_datatype", or needs guidance on metaxy feature definitions, versioning, metadata stores, CLI commands, testing patterns, feature locking, Map datatype configuration, or multi-environment configuration.
---

# Metaxy

Metaxy is a metadata layer for multimodal Data and ML pipelines that manages and tracks feature versions, dependencies, and data lineage across complex computational graphs.

## Core Concepts

### Feature Definitions

To define a feature, create a class inheriting from `mx.BaseFeature` with a `FeatureSpec` metaclass argument:

```python
import metaxy as mx


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="media/video",
        id_columns=["video_id"],
        fields=["audio", "frames"],  # Logical fields: describe data contents for versioning
    ),
):
    # Metadata columns: stored in the metadata store, tracked by metaxy
    video_id: str
    path: str
    duration: float
    height: int
    width: int
```

**Important distinction**: `fields` in `FeatureSpec` are logical field specs that describe the data contents for versioning and lineage tracking. Class attributes are metadata columns stored in the metadata store. They serve different purposes and should not overlap.

To add dependencies between features, use the `deps` parameter with `FeatureDep`. To specify field-level lineage (for partial data dependencies), use `FieldSpec` with `FieldDep` or `FieldsMapping`.

### Data Versioning

Metaxy automatically tracks sample versions and propagates changes through the dependency graph. To trigger recomputation when code changes, set `code_version` on `FieldSpec`:

```python
fields = [
    mx.FieldSpec(key="embedding", code_version="2"),  # Bump to invalidate downstream
]
```

### Metadata Stores

To configure a metadata store, create a `metaxy.toml` file or use programmatic configuration:

```python
config = mx.MetaxyConfig(
    stores={"dev": mx.StoreConfig(
        type="metaxy.ext.polars.DeltaMetadataStore",
        config={"root_path": "/tmp/metaxy"},
    )}
)
with config.use() as cfg:
    store = cfg.get_store("dev")
```

Supported backends: DuckDB, ClickHouse, BigQuery, LanceDB, Delta Lake.

### Feature Graph

To visualize and manage the feature dependency graph, use the CLI:

```bash
mx graph render            # Terminal visualization
mx push --store dev        # Push graph to store
```

## CLI

Metaxy provides a CLI (`metaxy` or `mx` alias) for managing features, metadata, and migrations:

```bash
mx list features --verbose     # List features with dependencies
mx graph render                # Visualize feature graph
mx metadata status --all-features  # Check metadata freshness (expensive!)
mx mcp                         # Start MCP server for AI assistants
```

## Multi-Environment & Feature Locking

Cross-project dependencies typically resolve automatically via Python packages — if project B depends on project A as a Python dependency, its features are discovered at import time. Feature locking (`mx lock`) is needed for multi-environment setups where projects cannot be installed into each other (e.g., separate deployment environments). In that case, use `mx push` to publish definitions to a shared store, and `mx lock` to fetch them into a `metaxy.lock` file. Set `locked = true` (or `METAXY_LOCKED=1`) in production to enforce version consistency.

For configuration patterns and CLI usage, see `examples/configuration.md` and `examples/cli.md`.

## Testing

To test features in isolation, use context managers to avoid polluting the global registry:

```python
import pytest
import metaxy as mx


@pytest.fixture
def metaxy_env(tmp_path):
    with mx.FeatureGraph().use():
        with mx.MetaxyConfig(
            stores={"test": mx.StoreConfig(
                type="metaxy.ext.polars.DeltaMetadataStore",
                config={"root_path": str(tmp_path / "delta_test")},
            )}
        ).use() as config:
            yield config
```

#### Map Datatype (Experimental)

To enable native Arrow Map column support (recommended for stores that support it), set `enable_map_datatype = true` in `metaxy.toml`. Requires the `polars-map` package. See https://docs.metaxy.io/stable/guide/concepts/metadata-stores/#map-datatype

When working with Map columns, use `metaxy.utils.collect_to_polars`, `metaxy.utils.collect_to_arrow`, or `metaxy.utils.switch_implementation_to_polars` to materialize or convert frames. These utilities preserve Map column types that would otherwise be lost with standard Narwhals backend conversions. See https://docs.metaxy.io/stable/guide/concepts/metadata-stores/#map-datatype

## Examples

For complete code examples, see:

- `examples/feature-definitions.md` - Feature classes with dependencies and field-level deps
- `examples/configuration.md` - TOML and programmatic configuration
- `examples/metadata-stores.md` - Store operations
- `examples/testing.md` - Test isolation patterns
- `examples/cli.md` - CLI command reference

## Documentation

For comprehensive documentation: https://docs.metaxy.io/stable/

Key pages:

- **Quickstart**: https://docs.metaxy.io/stable/guide/quickstart/quickstart/
- **Feature Definitions**: https://docs.metaxy.io/stable/guide/concepts/definitions/features/
- **Data Versioning**: https://docs.metaxy.io/stable/guide/concepts/versioning/
- **Metadata Stores**: https://docs.metaxy.io/stable/guide/concepts/metadata-stores/
- **Projects**: https://docs.metaxy.io/stable/guide/concepts/projects/
- **External Features**: https://docs.metaxy.io/stable/guide/concepts/definitions/external-features/
- **CLI Reference**: https://docs.metaxy.io/stable/reference/cli/
- **API Reference**: https://docs.metaxy.io/stable/reference/api/
