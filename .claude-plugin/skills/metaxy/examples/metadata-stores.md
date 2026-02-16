# Metadata Stores

See full documentation: https://anam-org.github.io/metaxy/main/guide/concepts/metadata-stores/

## Store Operations

```python
import metaxy as mx

with store:
    # Resolve what needs to be computed
    increment = store.resolve_update(MyFeature)
    # increment.new - new samples
    # increment.stale - samples with changed provenance

with store:
    # Read metadata
    result = store.read(MyFeature, current_only=True)
    df = result.collect().to_polars()

with store.open("w"):
    # Write metadata
    store.write(MyFeature, df)
```
