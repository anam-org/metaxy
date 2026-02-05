---
title: "Feature Definitions"
description: "Declarative feature definitions with Pydantic models."
---

# Feature System

Metaxy has a declarative and flexible feature system.
It has been inspired by [Dagster](https://dagster.io/)'s Software-Defined Assets and [Nix](https://nixos.org/).

Feature definitions do not describe **data** (1)
{ .annotate }

1. except the logical fields versions

--8<-- "data-vs-metadata.md"

Metaxy is responsible for providing correct **metadata** to users. Metaxy does not interact with **data** directly, the user is responsible for writing it, typically using **metadata** to identify sample locations in storage.

!!! tip "Keeping Historical Data"

    Include `metaxy_data_version` in your data path to avoid collisions between different versions of the same data sample.
    Doing this will ensure that newer samples are never written over older ones.

## Feature Definitions

To create a Metaxy feature, extend the [`BaseFeature`][metaxy.BaseFeature] class (1).
{ .annotate }

1. It's a [Pydantic](https://docs.pydantic.dev/latest/) model.


!!! abstract

    Features must have unique (across all projects) [`FeatureKey`][metaxy.FeatureKey] associated with them.

    Users must provide one or more ID columns (1) to [`FeatureSpec`][metaxy.FeatureSpec], telling Metaxy how to uniquely identify feature samples.
    { .annotate }

    1. ID columns are *almost* a primary key. The difference is quite subtle: Metaxy may interact with storage systems which do not technically have the concept of a primary key and may allow multiple rows to have the same ID columns (which are deduplicated by Metaxy).

```py
import metaxy as mx


class VideoFeature(mx.BaseFeature, spec=mx.FeatureSpec(key="raw/video", id_columns=["video_id"])):
    path: str
```

Since `VideoFeature` is a **root feature**, it doesn't have any dependencies.

That's it! Easy.

!!! question annotate "Why classes?"
    Some of the tooling Metaxy is aiming to integrate with, such as [SQLModel](/integrations/plugins/sqlmodel.md) or [Lance](/integrations/metadata-stores/databases/lancedb.md) are using class-based table definitions.
    It was practical to start from this interface, since it's somewhat more complicated to implement and support.
    More feature definition and registration methods are likely to be introduced in the future, since Metaxy doesn't
    use the class information in any way (1).

1. That's a little lie. The [Dagster integration](/integrations/orchestration/dagster/index.md) uses the original class to extract the table schema for visualization purposes, but we are exploring alternative solutions in [`anam-org/metaxy`](https://github.com/anam-org/metaxy/issues/855)

!!! tip

    You may now use `VideoFeature.spec()` class method to access the original feature spec: it's bound to the class.

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

## Field-Level Dependencies

A core (1) feature of Metaxy is the concept of **field-level dependencies**.
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
    For example, a `Video` feature - an `.mp4` file - may have `frames` and `audio` fields.

At this point, careful readers have probably noticed that the `Transcript` feature from the example above should not depend on the full video: it only needs the audio track in order to generate the transcript.
Let's express this with Metaxy:

```py
import metaxy as mx

video_spec = mx.FeatureSpec(key="raw/video2", id_columns=["video_id"], fields=["audio", "frames"])


class VideoFeature2(mx.BaseFeature, spec=video_spec):
    path: str


transcript_spec = mx.FeatureSpec(
    key="raw/transcript",
    id_columns=["video_id"],
    deps=[VideoFeature2],
    fields=[
        mx.FieldSpec(
            key="text",
            deps=[mx.FieldDep(feature=VideoFeature2, fields=["audio"])],
        )
    ],
)


class TranscriptFeature(mx.BaseFeature, spec=transcript_spec):
    path: str
```

Voilà!

> [!TIP] Use boilerplate-free API
> Metaxy allows passing simplified types to some of the models like `FeatureSpec` or `FeatureKey`.
> See [syntactic sugar](../syntactic-sugar.md) for more details.

The [Data Versioning](../data-versioning.md) docs explain more about how Metaxy calculates versions for different components of a feature graph.

## Attaching user-defined metadata

Users can [attach](/reference/api/definitions/feature-spec.md#metaxy.FeatureSpec.metadata) arbitrary JSON-like metadata dictionary to feature specs, typically used for declaring ownership, providing information to third-party tooling, or documentation purposes.
This metadata does not influence graph topology or the versioning system.

## Fully Qualified Field Key

!!! abstract

    A **fully qualified field key (FQFK)** is an identifier that uniquely identifies a field within the whole feature graph.

It consists of the **feature key** and the **field key**, separated by a colon.

!!! example

    -  `/raw/video:frames`

    -  `/raw/video:audio/english`

## External Features

External features are stubs pointing at features actually defined in other projects and not available in Python at runtime.
They can be used if the actual feature class cannot be imported, for example due to dependency conflicts or for other reasons.

Externals features can be defined with:

```py
import metaxy as mx

external_feature = mx.FeatureDefinition.external(
    spec=mx.FeatureSpec(key="a/b/c", id_columns=["id"]),
    project="external-project",
)
```

External features only exist until the actual feature definitions are loaded from the metadata store and replace them. This can be done with [`metaxy.sync_external_features`][metaxy.sync_external_features].

```python
import metaxy as mx

# Sync external features from the metadata store
mx.sync_external_features(store)
```

!!! note "Pydantic Schema Limitation"

    Features loaded from the metadata store have their JSON schema preserved from when they were originally saved. However, the Pydantic model class is not available. Operations that require the actual Python class, such as model instantiation or validation, will not work for these features.

Metaxy has a few safe guards in order to combat incorrect versioning information on external feature definitions. By default, Metaxy emits warnings when an external feature appears to have a different version (or field versions) than the actual feature definition loaded from the other project. These warnings can be turned into errors by:

- passing `on_conflict="raise"` to [`sync_external_features`][metaxy.sync_external_features]
- passing `--locked` to Metaxy CLI commands
- setting `locked` to `True` in the global Metaxy configuration. This can be done either in the config file or via the `METAXY_LOCKED` environment variable.

!!! tip

    We recommend setting `METAXY_LOCKED=1` in production

Additionally, the following actions always trigger a sync for external feature definitions:

- pushing feature definitions to the metadata store (e.g. `metaxy push` CLI)
- [`MetadataStore.read`][metaxy.MetadataStore.resolve_update]
- [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update]

And some other places where it's appropriate and doesn't create an additional overhead.

!!! tip

    This behavior can be disabled by setting `sync=False` in the global Metaxy configuration. However, we advise to keep it enabled,
    because [`sync_external_features`][metaxy.sync_external_features] is very lightweight on the first call and a no-op on subsequent calls.
    It only does anything if the current feature graph does not contain any external features.
