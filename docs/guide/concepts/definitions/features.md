---
title: "Feature Definitions"
description: "Declarative feature definitions with Pydantic models."
---

# Features

Metaxy has a declarative  feature system inspired by [Dagster](https://dagster.io/)'s Software-Defined Assets and [Nix](https://nixos.org/).

Metaxy is responsible for providing correct **metadata** to users. Metaxy does not interact with **data** directly, the user is responsible for writing it, typically using **metadata** to identify sample locations in storage.

--8<-- "data-vs-metadata.md"

!!! tip "Keeping Historical Data"

    Include `metaxy_data_version` in your data path to avoid collisions between different versions of the same data sample.
    Doing this will ensure that newer samples are never written over older ones.

## Feature Definitions

!!! tip "These examples make use of Metaxy's syntactic sugar."

To create a Metaxy feature, extend the [`BaseFeature`][metaxy.BaseFeature] class (1).
{ .annotate }

1. It's a [Pydantic](https://docs.pydantic.dev/latest/) model.

```py
import metaxy as mx


class VideoFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="raw/video", id_columns=["video_id"])):
    path: str
```

!!! abstract annotate

    Features must have unique (across all projects) [`FeatureKey`][metaxy.FeatureKey] associated with them.

    Users must provide one or more ID columns (1) to [`FeatureSpec`][metaxy.FeatureSpec], telling Metaxy how to uniquely identify feature samples.

1. ID columns are *almost* a primary key. The difference is quite subtle: Metaxy may interact with storage systems which do not technically have the concept of a primary key and may allow multiple rows to have the same ID columns (which are deduplicated by Metaxy).

Since `"raw/video"` is a **root feature**, it doesn't have any dependencies.

That's it! Easy.

!!! question annotate "Why classes?"
    Some of the tooling Metaxy is aiming to integrate with, such as [SQLModel](/integrations/plugins/sqlmodel.md) or [Lance](/integrations/metadata-stores/databases/lancedb.md) is using class-based table definitions.
    It was practical to start from this interface, since it's somewhat more complicated to implement and support.
    More feature definition and registration methods are likely to be introduced in the future, since Metaxy doesn't
    use the class information in any way (1).
    Additionally, users may want to construct instances of these Pydantic classes, and Pydantic can be used for data validation and type safety. We will explore other interfaces in [`anam-org/metaxy#800`](https://github.com/anam-org/metaxy/issues/800).

1. That's a little lie. The [Dagster integration](/integrations/orchestration/dagster/index.md) uses the original class to extract the table schema for visualization purposes, but we are exploring alternative solutions in [`anam-org/metaxy`](https://github.com/anam-org/metaxy/issues/855)

!!! tip

    You may now use `VideoFeature.spec()` class method to access the original feature spec: it's bound to the class.

### Read-Time Uniqueness

Use `FeatureSpec.unique` when multiple samples should appear once in the resolved
read view. To collapse equal values, group by a content digest:

```py
class DocumentBlob(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="documents/blobs",
        id_columns=["blob_id"],
        unique=mx.Unique(subset=["content_hash"]),
    ),
):
    blob_id: str
    content_hash: str
    path: str
```

When the digest should also define a field's data identity, provide it through
`metaxy_data_version_by_field`; Metaxy preserves caller-provided values on
write. See [Data Versioning](../versioning.md).

A content hash identifies a value: it changes when that value changes. An entity
key instead identifies something whose payload may change. Keep the latest state
for each entity by naming an explicit monotonic ordering column:

```py
class EntityBinding(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="registry/entity_bindings",
        id_columns=["entity_id", "revision"],
        unique=mx.Unique(
            subset=["entity_id"],
            keep="latest",
            order_by=["revision"],
        ),
    ),
):
    entity_id: str
    revision: int
    payload_digest: str
    status: str
```

For `keep="latest"`, the `order_by` columns followed by the feature's
`id_columns` must define a total logical order within each `subset` group. Rows
tied on that full key should be identical re-appends. Metaxy resolves conflicting
ties by physical metadata; rejecting them is currently the writer's
responsibility. NULL order values sort before non-NULL values.

Default reads first choose the physically current row for each `id_columns`
group. `order_by` only breaks a physical timestamp tie at this stage. When a
logical ordinal must dominate write order, include it in `id_columns`, as in
`EntityBinding`. Each revision then reaches the uniqueness step, where
`order_by` is the primary order.

Do not use a content hash as the entity key: changing the payload would silently
create a different entity. Re-appending an identical record leaves its
user-defined values unchanged in the resolved view, but system audit columns
may change: `metaxy_updated_at` is rewritten and
`metaxy_materialization_id` may differ. Those columns are not
re-append-stable watermarks.

Metaxy soft deletes and domain tombstones have different contracts. Default
current reads remove a Metaxy-soft-deleted sample before uniqueness. With
`include_soft_deleted=True`, deleted current rows participate normally and may
win. A domain tombstone is an ordinary highest-order record, such as
`status="tombstone"`, removed by a user filter after uniqueness:

<!-- skip next -->
```py
import narwhals as nw

store.read(
    EntityBinding,
    filters=[nw.col("status") != "tombstone"],
)
```

For downstream dependency resolution, place the same policy on the dependency
edge because direct read filters are not inherited:

<!-- skip next -->
```py
mx.FeatureDep(
    feature=EntityBinding,
    filters=["status != 'tombstone'"],
)
```

Pass `apply_unique=False` when unresolved rows are needed. Raw history for a
feature that is not registered in the current graph requires
`with_feature_history=True`, `with_sample_history=True`, and
`apply_unique=False`; Metaxy otherwise fails rather than silently skipping
uniqueness.

A latest-only retention process must keep Metaxy's current winner for every
`(id_columns, feature_version)` group, including the same tie and soft-delete
semantics. The default deduplicated view can be reconstructed from those rows;
retaining a naive `max(metaxy_updated_at)` row is not the contract. Explicit
history reads are outside this invariant.

Now let's define a child feature.

```py
class Transcript(
    mx.BaseFeature,
    spec=mx.FeatureSpec(key="processed/transcript", id_columns=["video_id"], deps=[VideoFeature]),
):
    transcript_path: str
    speakers_json_path: str
    num_speakers: int
```

??? abstract "The God `FeatureGraph` object"

    Features live on a global [`FeatureGraph`][metaxy.FeatureGraph] object (typically users do not need to interact with it directly).

Hurray! You get the idea.

### Field-Level Lineage

A core (1) feature of Metaxy is the concept of **field-level lineage**.
These are used to define dependencies between logical fields of features.
{ .annotate }

1. really a killer :gun:

!!! abstract

    A **Metaxy field** is not to be confused with **metadata column**.
    Columns refer to **metadata** and are stored in metadata stores (such as databases) supported by Metaxy. (1)
    { .annotate }

    1. columns can be defined with [**Pydantic fields**][pydantic.Field] :sweat_smile:

    Fields refer to **data** and are **purely logical** - users are free to define them as they see fit.
    Fields are supposed to represent parts of data that users care about.
    For example, a `"raw/video"` feature - an `.mp4` file - may have `frames` and `audio` fields.

At this point, careful readers have probably noticed that the `"processed/transcript"` feature from the example above should not depend on the full video: it only needs the audio track in order to generate the transcript.
Let's express this with Metaxy:

<!-- skip next -->
```py
class Transcript(
    mx.BaseFeature,
    spec=mx.FeatureSpec(key="processed/transcript", id_columns=["video_id"], fields=[
        mx.FieldSpec(
            key="text",
            deps=[mx.FieldDep(feature=VideoFeature, fields=["audio"])],
        )
    ],),
):
    transcript_path: str
    speakers_json_path: str
    num_speakers: int
```

The [Data Versioning](../versioning.md) docs explain more about how Metaxy calculates versions for different components of a feature graph.

### Attaching custom metadata

Users can [attach](/reference/api/definitions/feature-spec.md#metaxy.FeatureSpec.metadata) arbitrary JSON-like metadata dictionary to feature specs, typically used for declaring ownership, providing information to third-party tooling, or documentation purposes.
This metadata does not influence graph topology or the versioning system.

## Reusing Feature Definitions

It's often valuable to reuse the same base feature class across a few concrete feature definitions. To achieve this, set `spec` to `None`: Metaxy won't treat the class as a feature definition. For example:

<!-- skip next -->
```py
class MyFeatureBase(mx.BaseFeature, spec=None):
    ...


class MyDownstreamFeature(MyFeatureBase, spec=mx.FeatureSpec(...)):
    ...
```

This allows taking advantage of inheritance patterns for feature definitions, sharing the same set of metadata columns across them, and so on.
