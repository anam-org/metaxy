# Error Tracking

Metaxy provides first-class error tracking to gracefully handle failures during data processing.
When samples fail to process, errors are recorded and automatically excluded from downstream processing.
Further, persistent failures may be skipped from retrying.

Error tracking in Metaxy allows you to:

- **Record errors** for specific samples that fail during processing
- **Exclude failed samples** from downstream updates automatically
- **Track error history** with version-specific error isolation
- **Recover from failures** by reprocessing fixed data
- **Propagate error context** through feature dependency chains

## Core Concepts

### Error Tables

Each feature gets a corresponding error table with naming convention: `{feature_table_name}__errors`

Error tables:

- Store error messages, types, and associated sample IDs
- Include Metaxy system columns (`metaxy_feature_version`, `metaxy_snapshot_version`, `metaxy_created_at`)
- Are hidden from normal graph operations (don't appear in `graph.list_features()`)
- Use the same storage backend as feature tables
- Are automatically cleaned up when parent feature is dropped

### Error Exclusion

By default, `resolve_update()` excludes samples with recorded errors from the `added` frame:

- Only affects the `added` frame (new samples to process)
- Does not affect `changed` or `removed` frames
- Version-specific: only errors for the current `feature_version` cause exclusion
- Can be disabled with `exclude_errors=False`

### Automatic Error Clearing

When `write_metadata()` successfully writes samples, their errors are automatically cleared:

- Implements "success invalidates errors" pattern
- Only clears errors for the specific samples written
- Runs in a try/except block (won't fail the write if error clearing fails)
- Helps implement retry-after-fix workflows

## Recording Errors

### Manual Error Logging

Use the `catch_errors()` context manager with manual `log_error()` calls for fine-grained control:

```python
from metaxy import MetadataStore, BaseFeature, FeatureSpec, FeatureKey


class ImageFeature(
    BaseFeature,
    spec=FeatureSpec(
        key=FeatureKey(["images", "processed"]),
        id_columns=("image_id",),
    ),
):
    image_id: str
    width: int
    height: int


store = MetadataStore(...)

with store.open(mode="write"):
    with store.catch_errors(ImageFeature, autoflush=True) as ctx:
        for image in images:
            try:
                # Process image
                result = process_image(image)
                # Write successful result
                store.write_metadata(ImageFeature, result)
            except ValueError as e:
                # Log error for this specific sample
                ctx.log_error(
                    message=str(e), error_type="ValueError", image_id=image["id"]
                )
```

### Automatic Exception Catching

The `catch_errors()` context manager can automatically catch and record exceptions:

```python
with store.open(mode="write"):
    with store.catch_errors(ImageFeature, autoflush=True):
        # Any exception raised here is automatically caught and recorded
        result = risky_operation()
        store.write_metadata(ImageFeature, result)
```

**Note:** Automatically caught exceptions don't have sample ID context (all id_columns are None). For sample-specific error tracking, use manual `log_error()` calls.

### Collecting Errors for Batch Writing

Set `autoflush=False` to collect errors without writing them immediately:

```python
with store.open(mode="write"):
    # Collect errors without writing
    with store.catch_errors(ImageFeature, autoflush=False) as ctx:
        for image in images:
            try:
                result = process_image(image)
            except Exception as e:
                ctx.log_error(
                    message=str(e), error_type=type(e).__name__, image_id=image["id"]
                )

    # Access collected errors
    errors_df = store.collected_errors["images.processed"]

    # Optionally filter or transform
    errors_df = errors_df.filter(pl.col("error_type") != "Warning")

    # Write when ready
    store.write_errors(ImageFeature, errors_df)
```

### Writing Errors Directly

You can also construct and write error DataFrames directly:

```python
import polars as pl

errors_df = pl.DataFrame(
    {
        "image_id": ["img_001", "img_002"],
        "error_message": ["Invalid format", "File not found"],
        "error_type": ["ValueError", "FileNotFoundError"],
    }
)

with store.open(mode="write"):
    store.write_errors(ImageFeature, errors_df)
```

## Reading and Managing Errors

### Reading Errors

Read errors for a feature with optional filtering:

```python
# Read all errors
errors = store.read_errors(ImageFeature)

# Read errors for specific samples
errors = store.read_errors(
    ImageFeature, sample_uids=[{"image_id": "img_001"}, {"image_id": "img_002"}]
)

# Read errors for specific feature version
errors = store.read_errors(ImageFeature, feature_version="abc123")

# Read only latest error per sample (deduplication)
errors = store.read_errors(ImageFeature, latest_only=True)
```

### Checking for Errors

Quickly check if errors exist:

```python
# Check if feature has any errors
has_errors = store.has_errors(ImageFeature)

# Check if specific sample has errors
has_errors = store.has_errors(ImageFeature, sample_uid={"image_id": "img_001"})
```

### Clearing Errors

Clear errors manually when needed:

```python
# Clear all errors for a feature
store.clear_errors(ImageFeature)

# Clear errors for specific samples
store.clear_errors(
    ImageFeature, sample_uids=[{"image_id": "img_001"}, {"image_id": "img_002"}]
)

# Clear errors for specific feature version
store.clear_errors(ImageFeature, feature_version="old_version")
```

## Error Exclusion in resolve_update

### Default Behavior

By default, `resolve_update()` excludes samples with errors:

```python
# Samples with errors are automatically excluded from the added frame
increment = store.resolve_update(ImageFeature, samples=new_images)

# Only non-errored samples are in increment.added
print(f"Samples to process: {len(increment.added)}")
```

### Disabling Error Exclusion

To include errored samples (e.g., for reprocessing):

```python
# Include all samples, even those with errors
increment = store.resolve_update(ImageFeature, samples=new_images, exclude_errors=False)
```

### Error Exclusion is Version-Specific

Only errors for the current `feature_version` cause exclusion:

```python
# Version 1: Process samples, some fail
class ImageFeature_V1(...):
    code_version = "1"
    ...


# Errors recorded with feature_version matching V1


# Version 2: Update feature code
class ImageFeature_V2(...):
    code_version = "2"  # Changed
    ...


# Errors from V1 don't affect V2 processing
increment = store.resolve_update(ImageFeature_V2, samples=new_images)
# Previously failed samples are included (different version)
```

## Common Patterns

### Pattern 1: Error Recovery Workflow

```python
# Step 1: Initial processing with error tracking
with store.open(mode="write"):
    with store.catch_errors(ImageFeature, autoflush=True) as ctx:
        for image in batch_1:
            try:
                result = process_image(image)
                store.write_metadata(ImageFeature, result)
            except Exception as e:
                ctx.log_error(
                    message=str(e), error_type=type(e).__name__, image_id=image["id"]
                )

# Step 2: Identify failed samples
errors = store.read_errors(ImageFeature).collect()
failed_ids = errors["image_id"].to_list()
print(f"Failed samples: {failed_ids}")

# Step 3: Fix data and reprocess
fixed_images = load_and_fix_images(failed_ids)

with store.open(mode="write"):
    for image in fixed_images:
        result = process_image(image)  # Should succeed now
        store.write_metadata(ImageFeature, result)
        # Errors automatically cleared for successful samples

# Step 4: Verify recovery
remaining_errors = store.read_errors(ImageFeature).collect()
print(f"Remaining errors: {len(remaining_errors)}")
```

### Pattern 2: Pipeline with Error Propagation

```python
# Define pipeline: Input → Processed → Aggregated
class InputFeature(...):
    sample_id: str
    raw_data: str


class ProcessedFeature(...):
    sample_id: str
    processed_data: str
    # Depends on InputFeature


class AggregatedFeature(...):
    sample_id: str
    aggregated: float
    # Depends on ProcessedFeature


# Process with error handling at each stage
with store.open(mode="write"):
    # Stage 1: Input (some may fail)
    with store.catch_errors(InputFeature, autoflush=True) as ctx:
        for sample in raw_samples:
            try:
                result = process_input(sample)
                store.write_metadata(InputFeature, result)
            except Exception as e:
                ctx.log_error(
                    message=str(e), error_type=type(e).__name__, sample_id=sample["id"]
                )

    # Stage 2: Processed (only processes successful inputs)
    samples = store.resolve_update(ProcessedFeature)
    # Failed inputs are automatically excluded

    with store.catch_errors(ProcessedFeature, autoflush=True) as ctx:
        for sample in samples.added:
            try:
                result = process_stage2(sample)
                store.write_metadata(ProcessedFeature, result)
            except Exception as e:
                ctx.log_error(
                    message=str(e),
                    error_type=type(e).__name__,
                    sample_id=sample["sample_id"],
                )

    # Stage 3: Aggregated (only processes successful processed samples)
    samples = store.resolve_update(AggregatedFeature)
    for sample in samples.added:
        result = aggregate(sample)
        store.write_metadata(AggregatedFeature, result)
```

### Pattern 3: Monitoring and Alerting

```python
def check_error_rates(store: MetadataStore, feature: type[BaseFeature]) -> dict:
    """Monitor error rates for a feature."""

    # Get total samples processed
    metadata = store.read_metadata(feature).collect()
    total_samples = len(metadata)

    # Get current errors
    errors = store.read_errors(feature, latest_only=True).collect()
    error_count = len(errors)

    # Calculate error rate
    error_rate = error_count / total_samples if total_samples > 0 else 0

    # Group errors by type
    error_types = errors.group_by("error_type").agg(pl.count().alias("count")).collect()

    return {
        "total_samples": total_samples,
        "error_count": error_count,
        "error_rate": error_rate,
        "error_types": error_types.to_dicts(),
    }


# Use in monitoring
stats = check_error_rates(store, ImageFeature)
if stats["error_rate"] > 0.1:  # Alert if >10% error rate
    alert(f"High error rate detected: {stats['error_rate']:.1%}")
```

## Best Practices

### When to Use Error Tracking

**Use error tracking when:**

- Processing can fail for individual samples (data quality issues, parsing errors)
- You want to continue processing despite failures
- You need to track and monitor failure patterns
- You want to implement retry-after-fix workflows
- Downstream features should skip failed samples

**Don't use error tracking when:**

- Failures are systemic (code bugs, infrastructure issues)
- All samples should succeed or the entire batch fails
- Errors are expected to be extremely rare
- You prefer fail-fast behavior

### Error Messages

**Keep error messages concise:**

- Store error messages, not full stack traces (can be very large)
- Include relevant context (e.g., "Invalid value: -1 for field 'age'")
- Use consistent error types for easier grouping
- Consider extracting key information before storing

**Example:**

```python
try:
    value = parse_value(raw_data)
except ValueError as e:
    # Good: Concise with context
    ctx.log_error(
        message=f"Invalid value '{raw_data}': {str(e)[:200]}",
        error_type="ValueError",
        sample_id=sample["id"],
    )

    # Bad: Full stack trace (too large)
    # message=traceback.format_exc()
```

### Error Clearing Strategy

**Choose the right clearing strategy:**

1. **Automatic clearing (default)**: Errors cleared when samples succeed
   - Best for most use cases
   - Implements retry-after-fix naturally
   - No manual cleanup needed

2. **Manual clearing**: Explicitly call `clear_errors()`
   - Use when you want to preserve error history
   - Use for periodic cleanup of old errors
   - Use when implementing custom recovery logic

### Performance Considerations

**Optimize error tracking for large-scale processing:**

- Use `autoflush=False` for batch processing (write errors in bulk)
- Use `latest_only=True` when reading errors to deduplicate
- Periodically clean up old errors for old feature versions
- Consider error table size in storage capacity planning
- Error tables inherit the backend performance characteristics

## Limitations

### ID Columns Required

Error tracking requires features to have `id_columns` defined. Root features without ID columns cannot use error tracking.

### Backend Compatibility

When using `resolve_update()` with error exclusion, the samples DataFrame and error table should use compatible backends:

- **Polars samples + Polars error store**: Full support
- **Ibis samples + Ibis error store**: Full support
- **Mixed backends**: Falls back gracefully (logs warning, skips exclusion)

### Error Table Visibility

Error tables are hidden from:

- `graph.list_features()`
- Graph snapshots
- Migration operations

But error tables are visible in:

- Raw database queries
- Backend-specific table listings (e.g., `SHOW TABLES` in SQL)

## API Reference

For detailed API documentation, see:

- [`MetadataStore.catch_errors()`](../reference/api/metadata-stores/index.md#metaxy.MetadataStore.catch_errors)
- [`MetadataStore.write_errors()`](../reference/api/metadata-stores/index.md#metaxy.MetadataStore.write_errors)
- [`MetadataStore.read_errors()`](../reference/api/metadata-stores/index.md#metaxy.MetadataStore.read_errors)
- [`MetadataStore.clear_errors()`](../reference/api/metadata-stores/index.md#metaxy.MetadataStore.clear_errors)
- [`MetadataStore.has_errors()`](../reference/api/metadata-stores/index.md#metaxy.MetadataStore.has_errors)
- [`ErrorContext`](../reference/api/metadata-stores/index.md#metaxy.metadata_store.base.ErrorContext)
