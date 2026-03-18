---
title: "Tutorial"
description: "Step-by-step tutorial building a video ML pipeline with incremental recomputation."
---

# Tutorial

## Overview

::: metaxy-example source-link
    example: intro

This tutorial walks through building a multimodal ML pipeline that processes video files. By the end you will understand Metaxy's core workflow: declaring features with dependencies, resolving incremental updates, and detecting when code changes require selective recomputation.

The pipeline extracts faces and transcriptions from videos, then summarizes the transcriptions:

::: metaxy-example graph
    example: intro
    scenario: "Initial pipeline run"
    direction: LR

## Project Setup

Install the example's dependencies:

```shell
uv sync
```

A Metaxy project needs a `metaxy.toml` that tells Metaxy where to find feature definitions and which metadata store to use:

<!-- dprint-ignore-start -->
```toml title="metaxy.toml"
--8<-- "example-intro/metaxy.toml"
```
<!-- dprint-ignore-end -->

The `entrypoints` field points to a module that imports the feature definitions so Metaxy can discover them. The pipeline initializes Metaxy and obtains a store handle:

<!-- dprint-ignore-start -->
```python title="src/example_intro/pipeline.py"
--8<-- "example-intro/src/example_intro/pipeline.py:init"
```
<!-- dprint-ignore-end -->

## Defining Features

### `"video"` -- root feature

The root feature represents raw video files. It has no upstream dependencies — its provenance is provided externally via `metaxy_provenance_by_field` rather than derived from other features.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py"
--8<-- "example-intro/src/example_intro/features.py:video_feature"
```
<!-- dprint-ignore-end -->

### `"video/faces"` -- face detection

Depends on `"video"`. The `faces` field has a `code_version` that tracks the algorithm version.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py" hl_lines="4"
--8<-- "example-intro/src/example_intro/features.py:face_detection_feature"
```
<!-- dprint-ignore-end -->

The `deps=[Video]` declaration tells Metaxy that when the parent's provenance changes, face detection must be recomputed. Changes to sibling features like `"video/transcription"` do not affect this feature.

### `"video/transcription"` -- audio transcription

Also depends on `"video"`, but is independent of `"video/faces"`.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py" hl_lines="4"
--8<-- "example-intro/src/example_intro/features.py:audio_transcription_feature"
```
<!-- dprint-ignore-end -->

### `"video/summary"` -- summarization

Depends on `"video/transcription"`. When the transcription algorithm changes, summaries are automatically recomputed. Changes to `"video/faces"` do not trigger recomputation here because there is no dependency path between them.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py" hl_lines="4"
--8<-- "example-intro/src/example_intro/features.py:summary_feature"
```
<!-- dprint-ignore-end -->

## Running the Pipeline

### Ingesting root samples

Root features need explicit input samples. Each sample carries a `metaxy_provenance_by_field` column — a dictionary mapping field keys to their current data version. When a value changes, Metaxy knows the underlying data has been updated and downstream features need recomputation. The `"default"` key applies to all fields that do not have an explicit entry.

<!-- dprint-ignore-start -->
```python title="src/example_intro/pipeline.py"
--8<-- "example-intro/src/example_intro/pipeline.py:video_samples"
```
<!-- dprint-ignore-end -->

All reads and writes happen inside a [`store.open("w")`][metaxy.MetadataStore.open] context manager, which manages the store connection.

[`resolve_update`][metaxy.MetadataStore.resolve_update] compares the incoming provenance to the stored state and returns an [`Increment`][metaxy.Increment] with three partitions: `new` (unseen samples), `stale` (changed samples), and `orphaned` (removed samples). After processing, [`write`][metaxy.MetadataStore.write] records their metadata so they are not reprocessed next time.

### Processing downstream features

Non-root features do not need explicit samples. Metaxy resolves them automatically from the upstream metadata. The pattern is the same for every downstream feature: resolve, process, write.

<!-- dprint-ignore-start -->
```python title="src/example_intro/pipeline.py"
--8<-- "example-intro/src/example_intro/pipeline.py:face_detection"
```
<!-- dprint-ignore-end -->

The transcription and summary steps follow the same pattern.

## Walkthrough

### Step 1: Initial Run

Run the pipeline to process all videos through face detection, transcription, and summarization:

::: metaxy-example output
    example: intro
    scenario: "Initial pipeline run"
    step: "initial_run"

All four features were materialized for the 3 video samples.

### Step 2: Verify Idempotency

Run the pipeline again without any changes:

::: metaxy-example output
    example: intro
    scenario: "Idempotent rerun"
    step: "idempotent_run"

No recomputation occurred. Metaxy detected that all provenances match the stored state and skipped every feature.

### Step 3: Update Transcription Algorithm

Simulate an improved transcription model by changing `"video/transcription"`'s `code_version` from `"1"` to `"2"`:

::: metaxy-example patch-with-diff
    example: intro
    path: patches/01_update_transcription.patch
    scenario: "Code evolution"
    step: "update_transcription_version"

The existing transcriptions and the downstream summaries now need to be recomputed.

### Step 4: Observe Selective Recomputation

Run the pipeline again after the algorithm change:

::: metaxy-example output
    example: intro
    scenario: "Code evolution"
    step: "recompute_after_change"

`"video/transcription"` and its downstream `"video/summary"` were recomputed. `"video/faces"` was **not** recomputed because it depends only on `"video"`, which did not change.

## How It Works

The branching graph in this example has two independent paths from `"video"`:

- `"video"` → `"video/faces"` (face detection branch)
- `"video"` → `"video/transcription"` → `"video/summary"` (transcription branch)

When the transcription `code_version` changes, Metaxy recomputes the feature version for `"video/transcription"`. This changes its provenance, which propagates to `"video/summary"`. The face detection branch is unaffected because its version depends only on `"video"`, which did not change.

This is the core mechanism: [`resolve_update`][metaxy.MetadataStore.resolve_update] computes what the provenance _would be_ given the current definitions and compares it to what is stored. Only samples where these differ are returned for recomputation.

## What's Next?

- [Quickstart](../guide/quickstart/quickstart.md) -- deeper dive into the API: `Increment` fields, data versioning, and flushing metadata
- [Basic Example](basic.md) -- two-feature pipeline with explicit field-level dependencies via [`FeatureDep`][metaxy.FeatureDep]
- [Features and Fields](../guide/concepts/definitions/features.md) -- field-level lineage with [`FieldDep`][metaxy.FieldDep]
- [Data Versioning](../guide/concepts/versioning.md) -- how versions are calculated
- [Relationships](../guide/concepts/definitions/relationship.md) -- aggregation and expansion patterns
