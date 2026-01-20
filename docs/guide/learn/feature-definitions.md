---
title: "Feature Definitions"
description: "Declarative feature definitions with Pydantic models."
---

# Feature System

Metaxy has a declarative (defined statically at class level), expressive, flexible feature system.
It has been inspired by [Dagster](https://dagster.io/)'s Software-Defined Assets and [Nix](https://nixos.org/).

!!! abstract

    Features represent tabular **metadata**, typically containing references to external multi-modal **data** such as files, images, or videos.

    | **Subject** | **Description** |
    |---------|-------------|
    | **Data** | The actual multi-modal data itself, such as images, audio files, video files, text documents, and other raw content that your pipelines process and transform. |
    | **Metadata** | Information about the data, typically including references to where data is stored (e.g., object store keys) plus additional descriptive entries such as video length, file size, format, version, and other attributes. |

    As an edge case, Metaxy features may also be pure **metadata** without references to external **data**.

I will highlight **data** and **metadata** with bold so it really stands out.

Metaxy is responsible for providing correct **metadata** to users.

During incremental processing, Metaxy will automatically resolve added, changed and deleted **metadata** rows and calculate the right [sample versions](data-versioning.md) for them.

Metaxy does not interact with **data** directly, the user is responsible for writing it, typically using **metadata** to identify sample locations in storage.

!!! tip "Keeping Historical Data"

    Include `metaxy_data_version` in your data path to avoid collisions between different versions of the same data sample.
    Doing this will ensure that newer samples are never written over older ones.

I hope we can stop using bold for **data** and **metadata** from now on, hopefully we've made our point.

## Feature Definitions

Metaxy provides a [`BaseFeature`][metaxy.BaseFeature] class that can be extended to create user-defined features.
It's a [Pydantic](https://docs.pydantic.dev/latest/) model.

!!! abstract

    Features must have unique (across all projects) [`FeatureKey`][metaxy.FeatureKey] associated with them.

    Users must provide one or more ID columns (1) to `FeatureSpec`, telling Metaxy how to uniquely identify feature samples.
    { .annotate }

    1. ID columns are *almost* a primary key. The difference is quite subtle: Metaxy may interact with storage systems which do not technically have the concept of a primary key and may allow multiple rows to have the same ID columns (which are deduplicated by Metaxy).

```py
from metaxy import BaseFeature, FeatureSpec


class VideoFeature(
    BaseFeature, spec=FeatureSpec(key="/raw/video", id_columns=["video_id"])
):
    path: str
```

Since `VideoFeature` is a **root feature**, it doesn't have any dependencies.

That's it! Easy.

!!! tip

    You may now use `VideoFeature.spec()` class method to access the original feature spec: it's bound to the class.

Now let's define a child feature.

```py
class Transcript(
    BaseFeature,
    spec=FeatureSpec(
        key="/processed/transcript", id_columns=["video_id"], deps=[VideoFeature]
    ),
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
from metaxy import FieldDep, FieldSpec

video_spec = FeatureSpec(key="/raw/video", fields=["audio", "frames"])


class VideoFeature(BaseFeature, spec=video_spec):
    path: str


transcript_spec = TranscriptFeatureSpec(
    key="/raw/transcript",
    id_columns=["video_id"],
    fields=[
        FieldSpec(
            key="text",
            deps=[FieldDep(feature=VideoFeature, fields=["audio"])],
        )
    ],
)


class TranscriptFeature(BaseTranscriptFeature, spec=transcript_spec):
    path: str
```

Voilà!

> [!TIP] Use boilerplate-free API
> Metaxy allows passing simplified types to some of the models like `FeatureSpec` or `FeatureKey`.
> See [syntactic sugar](./syntactic-sugar.md) for more details.

The [Data Versioning](data-versioning.md) docs explain more about how Metaxy calculates versions for different components of a feature graph.

## Attaching user-defined metadata

Users can [attach](../../reference/api/definitions/feature-spec.md#metaxy.FeatureSpec.metadata) arbitrary JSON-like metadata dictionary to feature specs, typically used for declaring ownership, providing information to third-party tooling, or documentation purposes.
This metadata does not influence graph topology or the versioning system.

## Fully Qualified Field Key

!!! abstract

    A **fully qualified field key (FQFK)** is an identifier that uniquely identifies a field within the whole feature graph.

It consists of the **feature key** and the **field key**, separated by a colon.

!!! example

    -  `/raw/video:frames`

    -  `/raw/video:audio/english`
