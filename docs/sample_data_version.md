# Sample-Level Data Version Calculation

This document describes the sample-level data version calculation implementation using Polars expressions and the `polars-hash` plugin.

## Overview

The data version calculation is performed on a **sample level** (not on the feature/container level). Each sample gets its own unique data version based on:

1. The upstream data versions for that specific sample
2. The code version of the current container
3. A Merkle tree approach where all upstream dependencies are hashed together

## Key Functions

### `calculate_sample_data_versions()`

Calculates the data version for a single container across all samples in a DataFrame.

**Parameters:**
- `upstream_data_versions`: Dict mapping upstream feature keys to DataFrames with a `data_version` column (pl.Struct type)
- `container_key`: The key of the container being computed
- `code_version`: The code version of this container
- `container_deps`: Dict mapping upstream feature keys to lists of container keys this container depends on

**Returns:** A Polars expression that computes a struct with the data version for each sample.

**Example:**

```python
import polars as pl
from metaxy.data_version import calculate_sample_data_versions

# Upstream feature "video" has containers ["frames", "audio"]
upstream_df = pl.DataFrame({
    "sample_id": [1, 2, 3],
    "data_version": [
        {"frames": "abc123", "audio": "def456"},
        {"frames": "abc124", "audio": "def457"},
        {"frames": "abc125", "audio": "def458"},
    ]
})

upstream_data_versions = {"video": upstream_df}
container_deps = {"video": ["frames", "audio"]}

# Calculate data version for our "processed" container
expr = calculate_sample_data_versions(
    upstream_data_versions=upstream_data_versions,
    container_key="processed",
    code_version=1,
    container_deps=container_deps,
)

# Apply to get results
result = upstream_df.select([
    pl.col("sample_id"),
    expr.alias("data_version"),
])

# Extract the specific container's version
versions = result["data_version"].struct.field("processed")
```

### `calculate_feature_data_versions()`

Calculates data versions for **all containers** in a feature at once.

**Parameters:**
- `upstream_data_versions`: Dict mapping upstream feature keys to DataFrames
- `feature_containers`: Dict mapping container keys to their code versions
- `feature_deps`: Dict mapping container keys to their dependencies (dict of feature->containers)

**Returns:** A Polars expression that computes a struct with all container data versions.

**Example:**

```python
from metaxy.data_version import calculate_feature_data_versions

upstream_df = pl.DataFrame({
    "sample_id": [1, 2],
    "data_version": [
        {"frames": "abc", "audio": "def"},
        {"frames": "ghi", "audio": "jkl"},
    ]
})

upstream_data_versions = {"video": upstream_df}

feature_containers = {
    "processed": 1,    # code version 1
    "augmented": 2,    # code version 2
}

feature_deps = {
    "processed": {"video": ["frames"]},
    "augmented": {"video": ["frames", "audio"]},
}

expr = calculate_feature_data_versions(
    upstream_data_versions=upstream_data_versions,
    feature_containers=feature_containers,
    feature_deps=feature_deps,
)

result = upstream_df.select([
    pl.col("sample_id"),
    expr.alias("data_version"),
])

# Access individual container versions
processed_versions = result["data_version"].struct.field("processed")
augmented_versions = result["data_version"].struct.field("augmented")
```

## Data Structure

### Input: Upstream Data Versions

Each upstream feature provides a DataFrame with:
- A `sample_id` column (or other join key)
- A `data_version` column of type `pl.Struct`, where each field corresponds to a container

Example:
```python
pl.DataFrame({
    "sample_id": [1, 2, 3],
    "data_version": [
        {"container1": "hash1", "container2": "hash2"},
        {"container1": "hash3", "container2": "hash4"},
        {"container1": "hash5", "container2": "hash6"},
    ]
})
```

### Output: Computed Data Versions

The result is also a struct where each field is a container:

```python
pl.DataFrame({
    "sample_id": [1, 2, 3],
    "data_version": [
        {"processed": "newhash1", "augmented": "newhash2"},
        {"processed": "newhash3", "augmented": "newhash4"},
        {"processed": "newhash5", "augmented": "newhash6"},
    ]
})
```

## Hashing Algorithm

The data version is computed using SHA-256 via the `polars-hash` plugin:

1. Concatenate (with `|` separator):
   - Container key (e.g., `"processed"`)
   - Code version (e.g., `"1"`)
   - For each upstream dependency (sorted deterministically):
     - Feature/container identifier (e.g., `"video/frames"`)
     - The upstream data version hash for that container
2. Apply SHA-256 hash using `polars-hash`
3. Return the hex-encoded hash

This ensures:
- **Deterministic**: Same inputs always produce the same hash
- **Cascading**: Changes in any upstream container trigger recalculation
- **Version-aware**: Code version changes trigger recalculation
- **Vectorized**: Efficiently processes all samples in parallel

## Integration with FeaturePlan

The sample-level functions complement the existing `FeaturePlan` class in `models/plan.py`, which computes data versions at the feature/container level (scalar hashes). The sample-level functions enable per-sample tracking with the same Merkle tree approach but applied vectorized across all samples.
