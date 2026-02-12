---
title: "Hello, Metaxy!"
description: "Get started with using Metaxy."
---

# Quickstart

::: metaxy-example source-link
    example: quickstart

## First Metaxy Application

### 1. Install Metaxy

Let's choose a backend for our first [`MetadataStore`](/guide/concepts/metadata-stores.md).
A good option for local development is [DeltaLake](/integrations/metadata-stores/storage/delta.md).
Let's install it:

```shell
pip install 'metaxy[delta]'
```

Now the metadata store can be created as:

```py
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

store = DeltaMetadataStore("/tmp/quickstart.delta")
```

### 2. Define your first Feature

Any Metaxy project must have at least one root feature.

<!-- dprint-ignore-start -->
```python title="features.py"
--8<-- "example-quickstart/src/example_quickstart/features.py:video_feature"
```
<!-- dprint-ignore-end -->

### 3. Resolve a root increment

Root features are a bit special. They are entry points into the Metaxy world.
Because of that, will have to provide a `samples` argument to [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update],
which is typically not required for non-root features.

!!! tip annotate

    The only requirement for this dataframe is to have a `metaxy_provenance_by_field` column (1).

1. and to have appropriate ID columns

<!-- dprint-ignore-start -->
```py title="pipeline.py"
--8<-- "example-quickstart/src/example_quickstart/pipeline.py:resolve_video"
```
<!-- dprint-ignore-end -->

The `increment` object is an instance of [`Increment`][metaxy.Increment] and contains three dataframes:

- `increment.new`: new samples which were not previously recorded

- `increment.stale`: samples which were previously recorded but have now changed

- `increment.orphaned`: samples which were previously recorded but are no longer present in the input `samples` DataFrame

It's up to you how to handle these dataframes.
Usually there will be a processing step iterating over all the rows in `increment.new` and `increment.stale` (possibly in parallel, using something like [Ray](/integrations/compute/ray.md)), while `increment.orphaned` may be used to cleanup the no longer needed data and [metadata](/guide/concepts/deletions.md).

These dataframes have pre-computed provenance columns which **should not be modified** and eventually should be written to the metadata store.

!!! tip

    The dataframes will have a `metaxy_data_version` column which is recommended to be used for storage paths:

    <!-- dprint-ignore-start -->
    ```python title="pipeline.py"
    --8<-- "example-quickstart/src/example_quickstart/pipeline.py:process_video"
    ```
    <!-- dprint-ignore-end -->

### 4. Record metadata for processed samples

Once done, write the metadata for the processed samples:

<!-- dprint-ignore-start -->
```py title="pipeline.py"
--8<-- "example-quickstart/src/example_quickstart/pipeline.py:write_video"
```
<!-- dprint-ignore-end -->

Recorded samples will no longer be returned by `MetadataStore.resolve_update` during future pipeline runs, unless the incoming `metaxy_provenance_by_field` values are updated.

--8<-- "flushing-metadata.md"

## Feature Dependencies

Now let's add a downstream feature. We can use `deps` field on [`FeatureSpec`][metaxy.FeatureSpec] in order to do that. We will make a simple feature that extracts the audio track from a video.

<!-- dprint-ignore-start -->
```py title="features.py" hl_lines="5"
--8<-- "example-quickstart/src/example_quickstart/features.py:audio_feature"
```
<!-- dprint-ignore-end -->


And call the familiar `resolve_update` API:

<!-- dprint-ignore-start -->
```py title="pipeline.py"
--8<-- "example-quickstart/src/example_quickstart/pipeline.py:resolve_audio"
```
<!-- dprint-ignore-end -->

That's all! The increment can be handled similarly to the `Video` feature.

## Advanced Feature Definitions

Learn how to mark dependencies as optional, specify field-level versions and dependencies, lineage types, and other advanced Metaxy features in [definitions docs](/guide/concepts/definitions/features.md).

## What's Next?

Here are a few more useful links:

--8<-- "whats-next.md"
