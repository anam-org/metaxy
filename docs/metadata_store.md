# MetadataStore Tutorial

This guide introduces Metaxy's `MetadataStore` component, explaining its design philosophy, capabilities, and common usage patterns.

## What is a MetadataStore?

A `MetadataStore` is responsible for persisting and retrieving feature metadata - the information about your samples, their data versions, and any additional attributes you want to track.

Think of it as a versioned database for your feature pipeline, where each row represents a sample, and the metadata tracks which version of processing produced that sample.

## Core Concepts

### Metadata DataFrame

Metadata is represented as a Polars DataFrame with a specific structure:

```python
import polars as pl

metadata_df = pl.DataFrame({
    "sample_id": [1, 2, 3],
    "path": ["/data/sample1.mp4", "/data/sample2.mp4", "/data/sample3.mp4"],
    "data_version": [
        {"frames": "abc123", "audio": "def456"},
        {"frames": "abc124", "audio": "def457"},
        {"frames": "abc125", "audio": "def458"},
    ]
})
```

Key characteristics:
- **Required column**: `data_version` (pl.Struct type) - Maps container names to their version hashes
- **Sample identifiers**: Typically `sample_id` or similar columns to uniquely identify samples
- **User-defined columns**: Any additional metadata (paths, labels, timestamps, etc.)

### Immutability

**All metadata is immutable.** Once written, it never changes.

Why?
- **Reproducibility**: You can always recreate the exact same data
- **Caching**: Safe to cache aggressively since data never changes
- **Concurrency**: Multiple processes can write simultaneously without conflicts
- **Auditing**: Complete history is preserved

This means:
- ✅ Writing new metadata is always append-only
- ✅ Same data_version = same data, guaranteed
- ❌ No update or overwrite operations
- ❌ No delete operations (by design)

### Features and Containers

Metadata is organized by **features** (top-level entities) and **containers** (sub-components of features).

Example: A video processing feature might have:
```python
class VideoFeature(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "processing"]),
    containers=[
        ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
        ContainerSpec(key=ContainerKey(["audio"]), code_version=2),
    ]
)):
    pass
```

The `data_version` struct has one field per container:
```python
{"frames": "hash_of_frames_v1", "audio": "hash_of_audio_v2"}
```

## Design Philosophy

### Why Immutable?

In data pipelines, **reproducibility is paramount**. When you see a specific `data_version`, you should be able to:
1. Recreate the exact same output
2. Debug issues by looking at historical versions
3. Roll back to previous versions if needed
4. Share and compare results across environments

Immutability makes all of this trivial - there's no "current state" that might change, only a growing collection of versioned snapshots.

### Why Composable Stores?

Real-world workflows often span multiple environments:
- **Development**: Testing new features locally
- **Staging**: Validating changes before production
- **Production**: Running stable, battle-tested code

With composable stores, you can:
- Write to your local/development store
- Read unchanged features from production (via fallback)
- Only recompute what's actually changed
- Keep environments isolated (writes never affect upstream stores)

This is especially powerful for **branch deployments** where you're testing changes to one feature while reusing upstream features from production.

## Key Capabilities

### 1. Write Metadata

```python
from metaxy.metadata_store import InMemoryMetadataStore
import polars as pl

store = InMemoryMetadataStore()

# Prepare metadata with data_version
metadata = pl.DataFrame({
    "sample_id": [1, 2, 3],
    "path": ["/data/1.mp4", "/data/2.mp4", "/data/3.mp4"],
    "data_version": [
        {"frames": "abc123", "audio": "def456"},
        {"frames": "ghi789", "audio": "jkl012"},
        {"frames": "mno345", "audio": "pqr678"},
    ]
})

# Write to store (immutable, append-only)
store.write_metadata(VideoFeature, metadata)
```

### 2. Read Metadata

```python
# Read all metadata for a feature
df = store.read_metadata(VideoFeature)

# Read with filters
df = store.read_metadata(
    VideoFeature,
    filters=pl.col("sample_id") > 1
)

# Read specific columns only
df = store.read_metadata(
    VideoFeature,
    columns=["sample_id", "data_version"]
)
```

### 3. Check Feature Existence

```python
# Check if feature exists locally
if store.has_feature(VideoFeature):
    df = store.read_metadata(VideoFeature)

# Check including fallback stores
if store.has_feature(VideoFeature, check_fallback=True):
    df = store.read_metadata(VideoFeature, allow_fallback=True)
```

### 4. Fallback Store Chain

Create a store that reads from multiple sources:

```python
# Production store (read-only from dev's perspective)
prod_store = InMemoryMetadataStore()

# Development store with prod as fallback
dev_store = InMemoryMetadataStore(
    fallback_stores=[prod_store]
)

# This tries dev_store first, then prod_store
df = dev_store.read_metadata(VideoFeature, allow_fallback=True)

# This only checks dev_store
df = dev_store.read_metadata(VideoFeature, allow_fallback=False)
```

### 5. Upstream Dependency Resolution

Automatically load all upstream dependencies for a feature:

```python
class ProcessedVideoFeature(Feature, spec=FeatureSpec(
    key=FeatureKey(["processed", "video"]),
    deps=[FeatureDep(key=FeatureKey(["video", "processing"]))],
    containers=[
        ContainerSpec(
            key=ContainerKey(["default"]),
            code_version=1,
            deps=[ContainerDep(
                feature_key=FeatureKey(["video", "processing"]),
                containers=[ContainerKey(["frames"]), ContainerKey(["audio"])]
            )]
        )
    ]
)):
    pass

# Get all upstream metadata as dict
upstream_metadata = store.read_upstream_metadata(ProcessedVideoFeature)
# Returns: {"video_processing": <DataFrame with data_version column>}
```

### 6. Data Version Calculation

Automatically calculate data versions using upstream dependencies:

```python
# New samples to process (without data_version yet)
new_samples = pl.DataFrame({
    "sample_id": [4, 5, 6],
    "path": ["/data/4.mp4", "/data/5.mp4", "/data/6.mp4"],
})

# Calculate data versions based on upstream metadata and write
result_df = store.calculate_and_write_data_versions(
    feature=ProcessedVideoFeature,
    sample_df=new_samples,
    allow_upstream_fallback=True  # Can read upstream from fallback stores
)

# result_df now has data_version column calculated via Merkle tree hashing
print(result_df["data_version"])
```

## Computation Strategies

The `MetadataStore` supports two computation strategies for calculating data versions:

### 1. Backend-Native Computation (Optimized)

If the storage backend supports it AND all upstream dependencies are available locally, the store can compute data versions using native operations (e.g., SQL with SHA256 functions).

**Requirements:**
- Backend implements `_compute_data_versions_native()` method
- All upstream features exist in the same store (checked via `has_feature()`)
- No fallback to other stores needed

**Benefits:**
- Maximum performance (no data movement)
- Leverages database optimizations (parallelism, indexes)
- Memory efficient (computation happens in-database)

Example (conceptual - DuckDB backend):
```sql
-- Pure SQL computation with native SHA256
SELECT 
    new_samples.*,
    struct_pack(
        default := sha256_hex(
            concat('default|1|upstream_feature/frames|', 
                   upstream.data_version.frames)
        )
    ) as data_version
FROM new_samples
LEFT JOIN upstream_feature USING (sample_id)
```

### 2. Polars Computation (Universal Fallback)

If native computation isn't possible (backend doesn't support it OR upstream data is in fallback stores), the store falls back to Polars-based computation.

**Process:**
1. Load all upstream metadata into memory as Polars DataFrames
2. Use `polars-hash` plugin for SHA256 hashing
3. Compute data versions using Polars expressions
4. Write results back to store

**Benefits:**
- Works everywhere (universal fallback)
- Consistent hashing across all backends
- Handles cross-store scenarios gracefully

The strategy selection is **automatic and transparent** - you don't need to think about it!

```python
# This automatically chooses the best strategy:
# - Native if possible (all deps local, backend supports it)
# - Polars otherwise (deps in fallback stores or backend doesn't support native)
result = store.calculate_and_write_data_versions(
    feature=MyFeature,
    sample_df=new_samples,
)
```

## Common Usage Patterns

### Pattern 1: Simple Single-Store Usage

For local development or testing:

```python
from metaxy.metadata_store import InMemoryMetadataStore

store = InMemoryMetadataStore()

# Write metadata
store.write_metadata(MyFeature, metadata_df)

# Read it back
df = store.read_metadata(MyFeature)

# Check if feature exists
if store.has_feature(MyFeature):
    df = store.read_metadata(MyFeature)
else:
    # Compute and materialize
    df = compute_feature(MyFeature)
    store.write_metadata(MyFeature, df)
```

### Pattern 2: Branch Deployment Workflow

Testing changes while reusing production data:

```python
# Setup: Production store (already has FeatureA materialized)
prod_store = InMemoryMetadataStore()
# ... prod_store has FeatureA metadata ...

# Setup: Dev store with prod fallback
dev_store = InMemoryMetadataStore(fallback_stores=[prod_store])

# Scenario: Testing FeatureB (depends on FeatureA)
# FeatureA hasn't changed, so we can reuse from prod
# FeatureB has new code version, needs recomputation

# This will:
# 1. Check if all upstream (FeatureA) is in dev_store - NO
# 2. Fall back to Polars computation (loads FeatureA from prod_store)
# 3. Calculate FeatureB data_versions using Polars
# 4. Write FeatureB to dev_store only (prod remains unchanged)

new_feature_b_samples = pl.DataFrame({
    "sample_id": [1, 2, 3],
    "custom_field": ["a", "b", "c"],
})

result = dev_store.calculate_and_write_data_versions(
    feature=FeatureB,
    sample_df=new_feature_b_samples,
    allow_upstream_fallback=True  # Key: enables reading from prod
)

# Now dev_store has FeatureB, prod_store is untouched
assert dev_store.has_feature(FeatureB, check_fallback=False)  # True
assert prod_store.has_feature(FeatureB, check_fallback=False)  # False
```

### Pattern 3: Multi-Environment Pipeline

Chaining multiple environments:

```python
# Three-tier setup
prod_store = InMemoryMetadataStore()
staging_store = InMemoryMetadataStore(fallback_stores=[prod_store])
dev_store = InMemoryMetadataStore(fallback_stores=[staging_store])

# Dev can read from staging → prod
# Staging can read from prod
# Each writes only to itself

# Promotion workflow:
# 1. Develop and test in dev_store
# 2. Promote to staging_store (explicit copy)
# 3. Validate in staging
# 4. Promote to prod_store

# Promotion is just reading from dev and writing to staging
dev_metadata = dev_store.read_metadata(MyFeature, allow_fallback=False)
staging_store.write_metadata(MyFeature, dev_metadata)
```

### Pattern 4: Incremental Processing

Only process new samples by filtering existing metadata:

```python
# Get all previously processed samples
existing = store.read_metadata(MyFeature, columns=["sample_id"])
existing_ids = set(existing["sample_id"].to_list())

# Filter to only new samples
all_samples = pl.DataFrame({
    "sample_id": [1, 2, 3, 4, 5],
    "path": ["...", "...", "...", "...", "..."],
})

new_samples = all_samples.filter(
    ~pl.col("sample_id").is_in(existing_ids)
)

if len(new_samples) > 0:
    # Process and write only new samples
    result = store.calculate_and_write_data_versions(
        feature=MyFeature,
        sample_df=new_samples
    )
```

## Storage Backends

### InMemoryMetadataStore

Simple dict-based storage, useful for:
- Testing and prototyping
- Single-process applications
- Small datasets that fit in memory

```python
from metaxy.metadata_store import InMemoryMetadataStore

store = InMemoryMetadataStore()
```

**Limitations:**
- Data lost when process exits
- Not suitable for production
- No persistence across runs
- Uses Polars computation only (no native optimization)

### Future Backends (see roadmap.md)

The abstract `MetadataStore` interface enables future implementations:
- **Delta Lake**: Production-grade with time travel, ACID transactions
- **DuckDB**: High-performance analytical queries with native computation
- **ClickHouse**: Distributed analytics with SQL-based computation
- **SQLAlchemy**: Traditional database storage (generic SQL support)

Each backend can implement **backend-specific optimizations** like:
- **Native computation**: SHA256 in SQL instead of Polars
- **Partitioning**: Physical data layout (e.g., by `data_version`)
- **Indexing**: Fast lookups on specific columns
- **Query pushdown**: Execute filters at storage layer

## Partitioning Strategy

Partitioning is **backend-specific**, not part of the core API.

### Why Not in Core API?

1. Different backends have vastly different partition capabilities
2. Some backends don't support partitioning at all (in-memory)
3. Keeps the abstract interface simple and focused
4. Backends can auto-optimize based on query filters

### Common Approach

Most persistent backends will partition by `data_version` fields because:
- **Immutable**: Data versions never change
- **Natural boundary**: Each version is independent
- **Query pattern**: Often filtering by specific versions
- **Incremental writes**: New versions append naturally

Backend configuration example (future):
```python
# Backend configures partitioning, not core API
delta_store = DeltaMetadataStore(
    table_uri="s3://bucket/metadata/my_feature",
    partition_cols=["data_version"],  # Backend-specific config
)

# Reads automatically benefit from partition pruning
df = delta_store.read_metadata(
    MyFeature,
    filters=pl.col("data_version").struct.field("default") == "abc123"
)
# Backend automatically prunes to relevant partitions
```

## Best Practices

### 1. Design for Immutability

Structure your pipelines to produce immutable outputs:
```python
# ✅ Good: Generate new data_version for each run
new_data = process_samples(samples, code_version=2)
store.write_metadata(MyFeature, new_data)

# ❌ Bad: Don't try to "update" existing data
# (This isn't supported - data is immutable)
```

### 2. Use Feature Classes

Pass `Feature` classes instead of `FeatureKey` strings:
```python
# ✅ Good: Type-safe, auto-resolves dependencies
store.read_metadata(VideoFeature)

# ⚠️ Works but less convenient: Manual key construction
store.read_metadata(FeatureKey(["video", "processing"]))
```

### 3. Leverage Fallback Chains

Structure your deployment to reuse stable upstream data:
```python
# ✅ Good: Dev reads from prod, writes locally
dev_store = SomeStore(fallback_stores=[prod_store])

# ❌ Bad: Recomputing everything in dev
dev_store = SomeStore()  # No fallback, must compute all deps
```

### 4. Check Existence Before Computing

Use `has_feature()` to avoid unnecessary computation:
```python
# ✅ Good: Check before computing
if not store.has_feature(MyFeature):
    # Feature doesn't exist, compute it
    df = compute_expensive_feature()
    store.write_metadata(MyFeature, df)
else:
    # Feature exists, just read it
    df = store.read_metadata(MyFeature)

# ❌ Less efficient: Relying on exception handling
try:
    df = store.read_metadata(MyFeature)
except FeatureNotFoundError:
    df = compute_expensive_feature()
    store.write_metadata(MyFeature, df)
```

### 5. Filter Early

Use Polars expressions to filter at the storage layer:
```python
# ✅ Good: Filter pushed to storage
df = store.read_metadata(
    MyFeature,
    filters=pl.col("sample_id").is_in([1, 2, 3])
)

# ⚠️ Less efficient: Filtering after reading all data
df = store.read_metadata(MyFeature)
df = df.filter(pl.col("sample_id").is_in([1, 2, 3]))
```

### 6. Schema Validation

Ensure your DataFrames have the required `data_version` column:
```python
# The store will validate on write
try:
    store.write_metadata(MyFeature, df_without_data_version)
except MetadataSchemaError:
    # Add data_version column first
    df = calculate_data_versions(df_without_data_version)
    store.write_metadata(MyFeature, df)
```

## Error Handling

Common exceptions:

```python
from metaxy.metadata_store import (
    FeatureNotFoundError,
    MetadataSchemaError,
    DependencyError,
)

# Feature doesn't exist
try:
    df = store.read_metadata(MyFeature, allow_fallback=False)
except FeatureNotFoundError:
    # Compute and materialize
    df = compute_feature()
    store.write_metadata(MyFeature, df)

# Invalid schema
try:
    store.write_metadata(MyFeature, invalid_df)
except MetadataSchemaError as e:
    print(f"Schema validation failed: {e}")

# Missing upstream dependencies
try:
    result = store.calculate_and_write_data_versions(
        MyFeature,
        sample_df,
        allow_upstream_fallback=False
    )
except DependencyError as e:
    print(f"Missing upstream feature: {e}")
```

## What's Next?

- **API Reference**: Detailed method signatures and parameters
- **Roadmap** (`docs/roadmap.md`): Future features like MaterializationCatalog, additional backends
- **Examples**: Complete end-to-end pipeline examples
- **Advanced Topics**: Distributed computing, optimization strategies
