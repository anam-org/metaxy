# Dagster Integration Example

This example demonstrates how to integrate Metaxy with Dagster for orchestrating feature metadata pipelines.
They include metaxy specific progress logging and telemetry.

This example bundle includes:

- **Non-Partitioned** (`non_partitioned.py`): Process all samples at once.
- **Partitioned** (`partitioned.py`): Process one category at a time; each partition groups multiple videos, and each video has multiple rows (e.g., many frames).
- **Branch + Subsample** (`branch_subsampled.py`): Read from production fallback into a branch store, and subsample per run (random size or explicit keys).

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure Metaxy (see `metaxy.toml` for DuckDB configuration)

## Running the Examples

### Non-Partitioned Example

Process all videos at once:

```bash
# Start Dagster UI
dagster dev -m example_integration_dagster.non_partitioned

# Then materialize assets in the UI at http://localhost:3000
```

### Partitioned Example

Process one category at a time through the entire pipeline; each partition groups multiple videos, and each video carries multiple rows.

```bash
# Start Dagster UI
dagster dev -m example_integration_dagster.partitioned

# In the UI:
# 1. Add partitions for videos (e.g., "video_category_1", "video_category_2"); each category covers many videos/frames
# 2. Materialize each partition independently
```

### Branch + Subsample Example

Run against a branch store with production fallback. On each materialization, choose sampling via run config: random sample size or explicit keys.

```bash
# Start Dagster UI
dagster dev -m example_integration_dagster.branch_subsampled

# Example run config (random sample of 2):
# assets:
#   raw_video:
#     config:
#       sample_mode: random
#       sample_size: 2
#
# Example run config (explicit keys):
# assets:
#   raw_video:
#     config:
#       sample_mode: keys
#       sample_keys: ["video_123", "video_456"]
```

### Progress/Logging Example

Process in chunks with a tqdm-style progress bar (in TTY) and log updates:

```bash
dagster dev -m example_integration_dagster.progress_logging
```

## Key Concepts

### Root Features

- Must provide samples explicitly with `metaxy_provenance_by_field`
- For event-parallel, use `filter_samples_by_partition()` to filter samples

### Downstream Features

- Automatically load from upstream features
- For event-parallel, IOManager automatically filters to current partition
- For key-filtered runs, IOManager can be configured with `target_keys`/`target_key_column`

### Partitioned Processing

- Each partition represents one entity (video, document, user, etc.)
- Dagster orchestrates parallel execution across partitions
- One entity flows through entire pipeline: Raw → Clean → Embeddings

## Architecture

```
RawVideo (root)
    ↓
CleanVideo (downstream)
    ↓
VideoEmbeddings (downstream)
```

Each feature stores metadata in DuckDB, tracking:

- Feature versions
- Field dependencies
- Data provenance
- Incremental changes (added/changed/removed samples)
