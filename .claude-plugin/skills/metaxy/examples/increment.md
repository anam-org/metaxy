# Metadata Stores

See full documentation: https://anam-org.github.io/metaxy/main/guide/concepts/metadata-stores/

## Store Operations

```python
import metaxy as mx

with store:
    # Resolve what needs to be computed
    changes = store.resolve_update(MyFeature)
    # changes.new - new samples
    # changes.stale - samples with changed provenance

with store:
    # Read metadata
    result = store.read(MyFeature, current_only=True)
    df = result.collect().to_polars()

with store.open("w"):
    # Write metadata
    store.write_metadata(MyFeature, df)
```
