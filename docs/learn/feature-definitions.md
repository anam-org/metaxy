# Feature System

Metaxy has a declarative (defined statically at class level), expressive, flexible feature system. It has been inspired by Software-Defined Assets in [Dagster](https://dagster.io/).

Features represent tabular **metadata**, typically containing references to external multi-modal **data** such as files, images, or videos. But it can be just pure **metadata** as well.

I will highlight **data** and **metadata** with bold so it really stands out.

Metaxy is responsible for providing correct **metadata** to users. During incremental processing, Metaxy will automatically resolve added, changed and deleted **metadata** rows and calculate the right [sample versions](data-versioning.md) for them. Metaxy does not interact with **data** directly, the user is responsible for writing it, typically using **metadata** to identify sample locations in storage (it's a good idea to inject the sample version into the data sample identifier). Metaxy is designed to be used with systems that do not overwrite existing **metadata** (Metaxy only appends **metadata**) and therefore **data** as well (while we cannot enforce that since the user is responsible for writing the data, it's easily achievable by **including the sample version into the data sample identifier**).

I hope we can stop using bold for **data** and **metadata** from now on, hopefully we've made our point.

> [!tip] Include Sample Version In Your Data Path
> Include the sample version in your data path to ensure strong consistency guarantees. I mean it. Really do it!

Features live on a global `FeatureGraph` object (typically users do not need to interact with it directly). Features are bound to a specific Metaxy project, but can be moved between projects over time. Features must have unique (across all projects) `FeatureKey` associated with them.

## Feature Specs

Before we can define a `Feature`, we must first create a `FeatureSpec` object. But before we get to an example, it's necessary to understand the concept of ID columns. Metaxy must know how to uniquely identify feature samples and join metadata tables, therefore, you need to attach one or more ID columns to your `FeatureSpec`. Very often these ID columns would stay the same across many feature specs, therefore it makes a lot of sense to define them on a shared base class.

Some boilerplate with typing is involved (this is typically a good thing):

```py
from typing import TypeAlias

from metaxy import BaseFeatureSpec


VideoIds: TypeAlias = tuple[str]


class VideoFeatureSpec(BaseFeatureSpec[VideoIds]):
    id_columns: VideoIds = ("video_id",)
```

`BaseFeatureSpec` is a [Pydantic](https://docs.pydantic.dev/latest/) model, so all normal Pydantic features apply.

Feature specs now support an optional `metadata` dictionary for attaching ownership, documentation, or tooling context to a feature. This metadata **never** influences graph topology or version hashes, must be JSON-serializable, and is stored as an immutable [`frozendict`](https://pypi.org/project/frozendict/) once the spec is created (list values are frozen as tuples to guarantee immutability). It is ideal for values such as owners, SLAs, runbooks, or tags that external systems may want to inspect.

With our `VideoFeatureSpec` in place, we can proceed to defining features that would be using it.

## Feature Definitions

Metaxy provides a `BaseFeature` class that can be extended to make user-defined features. It's a Pydantic model as well. User-defined `BaseFeature` classes must have fields matching ID columns of the `FeatureSpec` they are using.

With respect to the same DRY principle, we can define a shared base class for features that use the `VideoFeatureSpec`.

```py
from metaxy import BaseFeature


class BaseVideoFeature(
    BaseFeature, spec=None
):  # spec=None is important to tell Metaxy this is a base class
    video_id: str
```

Now we are finally ready to define an actual feature.

```py
class VideoFeature(BaseVideoFeature, spec=VideoFeatureSpec(key="/raw/video")):
    path: str
```

That's it! That's a roow feature, it doesn't have any dependencies. Easy.

You may now use `VideoFeature.spec()` class method to access the original feature spec: it's bound to the class.

Now let's define a child feature.

```py
class Transcript(
    BaseVideoFeature,
    spec=VideoFeatureSpec(key="/processed/transcript", deps=[VideoFeature]),
):
    transcript_path: str
    speakers_json_path: str
    num_speakers: int
```

Hurray! You get the idea.

## Field-Level Dependencies

A core (I'be straight: a killer) feature of Metaxy is the concept of **field-level dependencies**. These are used to define dependencies between logical fields of features.

A **field** is not to be confused with metadata _column_ (Pydantic fields). Fields are completely independent from them.

Columns refer to _metadata_ and are stored in metadata stores (such as databases) supported by Metaxy.

Fields refer to _data_ and are logical -- users are free to define them as they see fit. Fields are supposed to represent parts of data that users care about. For example, a `Video` feature -- an `.mp4` file -- may have `frames` and `audio` fields.

Downstream features can depend on specific fields of upstream features. This enables fine-grained control over data versioning, avoiding unnecessary reprocessing.

At this point, careful readers have probably noticed that the `Transcript` feature from the [example](#feature-specs) above should not depend on the full video: it only needs the audio track in order to generate the transcript. Let's express that with Metaxy:

```py
from metaxy import FieldDep, FieldSpec

video_spec = VideoFeatureSpec(key="/raw/video", fields=[FieldSpec(key="audio"], FieldSpec(key="frames"))

class VideoFeature(BaseVideoFeature, spec=video_spec):
    path: str


transcript_spec = TranscriptFeatureSpec(key="/raw/transcript", fields=[FieldSpec(key="text", deps=[FieldDep(feature=VideoFeature.spec().key, fields=["audio"])])])

class TranscriptFeature(BaseTranscriptFeature, spec=transcript_spec):
    path: str
```

Voilà!

The [Data Versioning](data-versioning.md) docs explain more about this system.

### Fully Qualified Field Key

A **fully qualified field key (FQFK)** is an identifier that uniquely identifies a field within the whole feature graph. It consists of the **feature key** and the **field key**, separated by a colon, for example: `/raw/video:frames`, `/raw/video:audio/english`.

## A Note on Type Coercion for Metaxy types

Internally, Metaxy uses strongly typed Pydantic models to represent feature keys, their fields, and the dependencies between them.

To avoid boilerplate, Metaxy also has syntactic sugar for construction of these classes. Different ways to provide them are automatically coerced into canonical internal models. This is fully typed and only affects **constructor arguments**, so accessing **attributes** on Metaxy models will always return only the canonical types.

Some examples:

```py
from metaxy import FeatureKey

key = FeatureKey("prefix/feature")
key = FeatureKey(["prefix", "feature"])
key = FeatureKey("prefix", "feature")
same_key = FeatureKey(key)
```

Metaxy really loves you, the user! See [syntactic sugar](#syntactic-sugar) for more details.

## Syntactic Sugar

### Keys

Both `FeatureKey` and `FieldKey` accept:

- **String format**: `FeatureKey("prefix/feature")`
- **Sequence format**: `FeatureKey(["prefix", "feature"])`
- **Variadic format**: `FeatureKey("prefix", "feature")`
- **Same type**: `FeatureKey(another_feature_key)` -- for full Inception mode

All formats produce equivalent keys, internally represented as a sequence of parts
