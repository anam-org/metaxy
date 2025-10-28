# Video Chunking Example - One-to-Many Relationships in Metaxy

This example demonstrates how to handle **one-to-many relationships** in Metaxy, where a single parent feature (e.g., a video) produces multiple child features (e.g., video chunks).

## Overview

One-to-many relationships are common in ML pipelines:
- **Video → Chunks**: Split videos into temporal segments
- **Document → Pages**: Split documents into pages
- **Audio → Windows**: Split audio into overlapping windows
- **Image → Patches**: Extract patches from images

## Files

- `features.py`: Basic video chunking example with `load_input()` implementation
- `advanced_features.py`: Sophisticated example with dynamic chunking and aggregation
- `/src/metaxy/utils/one_to_many.py`: Utility functions for one-to-many patterns

## Key Concepts

### 1. Parent-Child Relationships

When implementing one-to-many relationships, you need to:
1. Generate unique child IDs deterministically
2. Maintain parent references for traceability
3. Handle joins correctly (child features may have different ID columns)

### 2. Sample UID Generation

Child sample UIDs must be:
- **Unique**: Each child needs a distinct identifier
- **Deterministic**: Same parent + index always produces same UID
- **Traceable**: Should maintain reference to parent

```python
from metaxy.utils.one_to_many import generate_child_sample_uid

# Generate deterministic child UID
child_uid = generate_child_sample_uid(
    parent_uid=12345,
    child_index=0,
    namespace="chunk"  # Avoid collisions between different child types
)
```

### 3. Custom `load_input()` Method

Override `load_input()` to implement fan-out logic:

```python
class VideoChunk(Feature, spec=...):
    @classmethod
    def load_input(cls, joiner, upstream_refs):
        from metaxy.utils.one_to_many import expand_to_children

        # Expand parent to children
        expanded = expand_to_children(
            parent_df=upstream_refs["video/raw"],
            num_children_per_parent=10,
            parent_ref_column="parent_video_id",
            namespace="chunk"
        )

        # Update refs and continue with standard joining
        expanded_refs = dict(upstream_refs)
        expanded_refs["video/raw"] = expanded

        return joiner.join_upstream(expanded_refs, cls.spec, ...)
```

## Usage Patterns

### Pattern 1: Fixed Fan-out
Each parent produces a fixed number of children:

```python
# In VideoChunk.load_input()
expanded = expand_to_children(
    parent_df,
    num_children_per_parent=10,  # Fixed 10 chunks per video
    parent_ref_column="parent_video_id"
)
```

### Pattern 2: Dynamic Fan-out
Number of children varies per parent:

```python
# Calculate chunks based on video duration
chunks_per_video = {
    video1_id: 5,   # 50 second video → 5 chunks
    video2_id: 12,  # 120 second video → 12 chunks
}

expanded = expand_to_children(
    parent_df,
    num_children_per_parent=chunks_per_video,
    parent_ref_column="parent_video_id"
)
```

### Pattern 3: Multiple Dependencies
When child depends on multiple parents with different semantics:

```python
class VideoChunk(Feature, spec=FeatureSpec(
    deps=[
        FeatureDep(key=Video.key),           # To expand
        FeatureDep(key=ChunkingStrategy.key) # Per-video config
    ],
    ...
)):
    @classmethod
    def load_input(cls, joiner, upstream_refs):
        # Join parents first
        video_with_strategy = video_ref.join(
            strategy_ref,
            on="video_id",  # Parent-level join
            how="inner"
        )

        # Then expand based on strategy
        # ... expansion logic ...
```

### Pattern 4: Aggregation (Many-to-One)
Aggregate child features back to parent level:

```python
class VideoSummary(Feature, spec=...):
    @classmethod
    def load_input(cls, joiner, upstream_refs):
        chunks = upstream_refs["video/chunks"]

        # Aggregate chunks back to video level
        aggregated = (
            chunks
            .group_by("parent_video_id")
            .agg([
                nw.col("score").mean().alias("avg_score"),
                nw.col("count").sum().alias("total_count"),
            ])
            .rename({"parent_video_id": "video_id"})
        )

        return aggregated, column_mapping
```

## Best Practices

### 1. Use Deterministic Child UIDs
Always use `generate_child_sample_uid()` or similar deterministic method:
```python
# Good - deterministic
child_uid = generate_child_sample_uid(parent_uid, index, namespace="chunk")

# Bad - random
child_uid = random.randint(0, 1000000)
```

### 2. Preserve Parent References
Always maintain parent references for traceability:
```python
expanded = expand_to_children(
    parent_df,
    num_children_per_parent=10,
    parent_ref_column="parent_video_id"  # Keep parent reference
)
```

### 3. Handle ID Column Changes
Be explicit when child features use different ID columns:
```python
spec=FeatureSpec(
    id_columns=["chunk_uid"],  # Different from parent's ["video_id"]
    ...
)
```

### 4. Document the Relationship
Always document the one-to-many semantics in docstrings:
```python
class VideoChunk(Feature, ...):
    """Individual video chunks (one-to-many from Video).

    Each Video produces N chunks where N depends on video duration.
    Chunks maintain parent_video_id reference.
    """
```

## Framework Improvements

The framework provides utilities to simplify one-to-many patterns:

1. **`expand_to_children()`**: Handles the mechanical aspects of fan-out
2. **`generate_child_sample_uid()`**: Creates deterministic child IDs
3. **`OneToManyJoiner`**: Experimental joiner for automatic expansion (see `one_to_many.py`)

Future improvements could include:
- Declarative one-to-many configuration in `FeatureSpec`
- Automatic parent reference tracking
- Built-in aggregation patterns
- Validation of parent-child relationships

## Running the Example

```bash
# Navigate to example directory
cd examples/src/examples/video_chunking

# List features to see the dependency graph
uv run metaxy list features

# The features demonstrate:
# - Video (root) → VideoChunk (one-to-many)
# - VideoChunk → DetectedFaces (one-to-one on chunks)
# - ChunkAnalysis → VideoSummary (many-to-one aggregation)
```

## Common Issues and Solutions

### Issue 1: Join Key Mismatch
**Problem**: Parent uses `video_id`, child uses `chunk_uid`
**Solution**: Use custom `load_input()` to handle different ID columns

### Issue 2: Lost Parent Data
**Problem**: After expansion, parent columns are missing
**Solution**: `expand_to_children()` preserves all parent columns by default

### Issue 3: Non-Deterministic UIDs
**Problem**: Different UIDs generated on each run
**Solution**: Use `generate_child_sample_uid()` with consistent namespace

### Issue 4: Duplicate Column Names
**Problem**: Multiple upstreams have same column names
**Solution**: Use `FeatureDep(columns=..., rename=...)` to select/rename

## Advanced Topics

### Custom Expansion Logic
For complex scenarios, implement custom expansion:

```python
def custom_expand(parent_df):
    # Custom logic for variable-length sequences,
    # overlapping windows, hierarchical splits, etc.
    ...
    return expanded_df
```

### Hierarchical Relationships
For multi-level hierarchies (Video → Scenes → Shots → Frames):

```python
# Each level implements its own load_input()
# maintaining references up the hierarchy
class Scene(Feature, ...):
    # video_id, scene_id

class Shot(Feature, ...):
    # video_id, scene_id, shot_id

class Frame(Feature, ...):
    # video_id, scene_id, shot_id, frame_id
```

### Cross-Product Relationships
For features that combine multiple one-to-many sources:

```python
class FramePairSimilarity(Feature, ...):
    # Compares all pairs of frames within a video
    # Requires custom load_input() for cross-product
```