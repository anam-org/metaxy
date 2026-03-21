---
title: "Ray Example"
description: "Distributed video processing with Ray Data, Dagster, and DuckLake."
---

# Ray

## Overview

::: metaxy-example source-link
    example: ray

Distributed video processing with Ray Data, Dagster, and DuckLake. Three progressive examples:

| Example | Orchestration | Highlights |
|---------|--------------|------------|
| 1. Ray Basics | Python script | `MetaxyDatasource`, `MetaxyDatasink`, incremental reads |
| 2. Ray + Dagster | Dagster assets | `@metaxify`, per-row writes via `BufferedMetadataWriter` |
| 3. Production Patterns | Dagster assets | Branching IO, run modes, error handling, counters |

::: metaxy-example graph
    example: ray
    scenario: "Setup video metadata"
    direction: LR

## Getting Started

```shell
uv sync
```

## Configuration

DuckLake-backed DuckDB metadata store with SQLite catalog and local file storage:

```toml title="metaxy.toml"
--8<-- "example-ray/metaxy.toml"
```

## Feature Definitions

Four features with field-level lineage. The `processing_time_seconds`, `tokens_used`, and `error` columns are part of the feature schema, making them queryable for operational monitoring.

<!-- dprint-ignore-start -->
```python title="src/example_ray/features.py"
--8<-- "example-ray/src/example_ray/features.py:features"
```
<!-- dprint-ignore-end -->

## Example 1: Ray Basics

```shell
python -m example_ray.basic_pipeline
```

Reads incremental samples via `MetaxyDatasource`, processes with `.map()`, and writes results via `MetaxyDatasink`. A second run shows idempotency.

<!-- dprint-ignore-start -->
```python title="src/example_ray/basic_pipeline.py"
--8<-- "example-ray/src/example_ray/basic_pipeline.py:basic_pipeline"
```
<!-- dprint-ignore-end -->

!!! tip

    Always pass `config` explicitly to `MetaxyDatasource` and `MetaxyDatasink` so Ray workers can discover the Metaxy configuration.

## Example 2: Ray + Dagster

```shell
dagster dev -m example_ray.definitions
```

Ray Data processing inside Dagster assets, following the `defs/` component structure:

- `video_metadata` -- seeds root video samples
- `crop_metadata` -- resolves increment from video, produces crop metadata
- `speech_to_text` -- resolves increment, processes with Ray, per-row writes via `BufferedMetadataWriter`
- `face_detection` -- same pattern for face detection from cropped frames

<!-- dprint-ignore-start -->
```python title="src/example_ray/defs/assets/__init__.py"
--8<-- "example-ray/src/example_ray/defs/assets/__init__.py:stt_asset"
```
<!-- dprint-ignore-end -->

Dagster definitions:

<!-- dprint-ignore-start -->
```python title="src/example_ray/definitions.py"
--8<-- "example-ray/src/example_ray/definitions.py:definitions"
```
<!-- dprint-ignore-end -->

!!! warning

    `BufferedMetadataWriter` runs a background thread. Always use it as a context manager to flush pending batches on exit.

## Example 3: Operational Patterns

```shell
dagster dev -m example_ray.production_definitions
```

Production patterns in `defs/assets/production.py`: branching IO, run modes, error handling, and operational counters. The production asset resolves updates for both `SpeechToText` and `Crop`, processes samples through Ray, and writes results to both features. This example uses separate definitions from Example 2 since the `video_processing` asset replaces `speech_to_text` and `crop_metadata` with a single branching asset.

### Run Modes

<!-- dprint-ignore-start -->
```python title="src/example_ray/defs/ops/run_config.py"
--8<-- "example-ray/src/example_ray/defs/ops/run_config.py:run_config"
```
<!-- dprint-ignore-end -->

| Mode | Use case |
|------|----------|
| `FULL` | Process all new and stale samples. |
| `KEYED` | Reprocess specific samples by UID. |
| `SUBSAMPLE` | Process a random fraction for smoke tests. |

### Error Handling

`FailurePolicy` defines acceptable failure thresholds. `timed_worker` captures per-record timing and error information:

<!-- dprint-ignore-start -->
```python title="src/example_ray/defs/ops/error_handling.py"
--8<-- "example-ray/src/example_ray/defs/ops/error_handling.py:failure_policy"
```
<!-- dprint-ignore-end -->

<!-- dprint-ignore-start -->
```python title="src/example_ray/defs/ops/worker.py"
--8<-- "example-ray/src/example_ray/defs/ops/worker.py:timed_worker"
```
<!-- dprint-ignore-end -->

### Pipeline Counters

`PipelineCounters` tracks operational metrics and surfaces them as Dagster output metadata:

<!-- dprint-ignore-start -->
```python title="src/example_ray/defs/ops/counters.py"
--8<-- "example-ray/src/example_ray/defs/ops/counters.py:counters"
```
<!-- dprint-ignore-end -->

### Production Asset

Combines branching IO, run modes, error handling, and counters. Results are joined with feature-specific provenance and written to both `SpeechToText` and `Crop` via `store.write`:

<!-- dprint-ignore-start -->
```python title="src/example_ray/defs/assets/production.py"
--8<-- "example-ray/src/example_ray/defs/assets/production.py:production_asset"
```
<!-- dprint-ignore-end -->

## Related Materials

- [Ray Integration](../integrations/compute/ray.md)
- [Dagster Integration](../integrations/orchestration/dagster/index.md)
- [DuckLake Storage](../integrations/metadata-stores/storage/ducklake.md)
- [DuckLake Example](ducklake.md)
