# Feature System

Metaxy has a declarative (defined statically at class level), expressive, flexible feature system.
It has been inspired by [Dagster](https://dagster.io/)'s Software-Defined Assets and [Nix](https://nixos.org/).

Features represent tabular **metadata**, typically containing references to external multi-modal **data** such as files, images, or videos.
But it can be just pure **metadata** as well.

I will highlight **data** and **metadata** with bold so it really stands out.

Metaxy is responsible for providing correct **metadata** to users.
During incremental processing, Metaxy will automatically resolve added, changed and deleted **metadata** rows and calculate the right [sample versions](data-versioning.md) for them.
Metaxy does not interact with **data** directly, the user is responsible for writing it, typically using **metadata** to identify sample locations in storage (it's a good idea to inject the sample version into the data sample identifier).
Metaxy is designed to be used with systems that do not overwrite existing **metadata** (Metaxy only appends **metadata**) and therefore **data** as well (while we cannot enforce that since the user is responsible for writing the data, it's easily achievable by **including the sample version into the data sample identifier**).

I hope we can stop using bold for **data** and **metadata** from now on, hopefully we've made our point.

> [!tip] Include sample version in your data path
> Include the sample version in your data path to ensure strong consistency guarantees.
> I mean it.
> Really do it!

Features live on a global `FeatureGraph` object (typically users do not need to interact with it directly).
Features are bound to a specific Metaxy project, but can be moved between projects over time.
Features must have unique (across all projects) `FeatureKey` associated with them.

## Feature Definitions

Metaxy provides a [`BaseFeature`][metaxy.BaseFeature] class that can be extended to create user-defined features.
It's a [Pydantic](https://docs.pydantic.dev/latest/) model.

```py
from metaxy import BaseFeature, FeatureSpec


class VideoFeature(
    BaseFeature, spec=FeatureSpec(key="/raw/video", id_columns=["video_id"])
):
    path: str
```

Metaxy must know how to uniquely identify feature samples and join metadata tables, therefore, you need to attach one or more ID columns to your `FeatureSpec`.

That's it!
Since it's a root feature, it doesn't have any dependencies.
Easy.

You may now use `VideoFeature.spec()` class method to access the original feature spec: it's bound to the class.

Now let's define a child feature.

```py
class Transcript(
    BaseFeature,
    spec=FeatureSpec(key="/processed/transcript", id_columns=["video_id"] deps=[VideoFeature]),
):
    transcript_path: str
    speakers_json_path: str
    num_speakers: int
```

Hurray! You get the idea.

## Field-Level Dependencies

A core (I'll be straight: a killer) feature of Metaxy is the concept of **field-level dependencies**.
These are used to define dependencies between logical fields of features.

A **field** is not to be confused with metadata _column_ (Pydantic fields).
Fields are completely independent from them.

Columns refer to _metadata_ and are stored in metadata stores (such as databases) supported by Metaxy.

Fields refer to _data_ and are logical -- users are free to define them as they see fit.
Fields are supposed to represent parts of data that users care about.
For example, a `Video` feature -- an `.mp4` file -- may have `frames` and `audio` fields.

Downstream features can depend on specific fields of upstream features.
This enables fine-grained control over field provenance, avoiding unnecessary reprocessing.

At this point, careful readers have probably noticed that the `Transcript` feature from the example above should not depend on the full video: it only needs the audio track in order to generate the transcript.
Let's express that with Metaxy:

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
            deps=[FieldDep(feature=VideoFeature.spec().key, fields=["audio"])],
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

Users can [attach](../reference/api/definitions/feature-spec.md#metaxy.FeatureSpec.metadata) arbitrary JSON-like metadata dictionary to feature specs, typically used for declaring ownership, providing information to third-party tooling, or documentation purposes.
This metadata does not influence graph topology or the versioning system.

### Fully Qualified Field Key

A **fully qualified field key (FQFK)** is an identifier that uniquely identifies a field within the whole feature graph.
It consists of the **feature key** and the **field key**, separated by a colon, for example: `/raw/video:frames`, `/raw/video:audio/english`.
