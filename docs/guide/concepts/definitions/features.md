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

Since `VideoFeature` is a **root feature**, it doesn't have any dependencies.

That's it! Easy.

!!! question annotate "Why classes?"
    Some of the tooling Metaxy is aiming to integrate with, such as [SQLModel](/integrations/plugins/sqlmodel.md) or [Lance](/integrations/metadata-stores/databases/lancedb.md) are using class-based table definitions.
    It was practical to start from this interface, since it's somewhat more complicated to implement and support.
    More feature definition and registration methods are likely to be introduced in the future, since Metaxy doesn't
    use the class information in any way (1).
    Additionally, users may want to construct instances of these Pydantic classes, and Pydantic can be used for data validation and type safety.

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

### Field-Level Dependencies

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

Voilà!


The [Data Versioning](../data-versioning.md) docs explain more about how Metaxy calculates versions for different components of a feature graph.

### Attaching custom metadata

Users can [attach](/reference/api/definitions/feature-spec.md#metaxy.FeatureSpec.metadata) arbitrary JSON-like metadata dictionary to feature specs, typically used for declaring ownership, providing information to third-party tooling, or documentation purposes.
This metadata does not influence graph topology or the versioning system.
