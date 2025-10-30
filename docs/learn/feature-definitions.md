# Feature System

Metaxy has a declarative (defined statically on class level), expressive, flexible feature system. It has been inspired by Software-Defined Assets in [Dagster](https://dagster.io/).

Features represent tabular **metadata**, typically containing references to multi-modal data such as files, images, or videos. But it can be pure metadata well.

Features live on a global `FeatureGraph` object (typically users do not need to interact with it directly). Features are bound to a specific Metaxy project, but can be moved between projects over time. Features must have unique (across all projects) `FeatureKey` associated with them.

## Feature Specs

Before we can define a `Feature`, we must first create a `FeatureSpec` object. But before we get to an example, it's necessary to understand the concept of ID columns. Metaxy must know how to uniquely identify feature samples and join metadata tables, therefore, you need to attach one or more ID columns to your `FeatureSpec`. Very often these ID columns would stay the same across many feature specs, therefore it makes a lot of sense to define them on a shared base class. Let's do it:

```py
from metaxy import BaseFeatureSpec


class VideoFeatureSpec(
    BaseFeatureSpec
):  # spec=None is important to tell Metaxy that this is a base class
    id_columns: tuple[str] = ("video_id",)
```

`BaseFeatureSpec` is a [Pydantic](https://docs.pydantic.dev/latest/) model, so all normal Pydantic features apply.

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


transcript_spec = TranscriptFeatureSpec(key="/raw/transcript", fields=[FieldSpec(key="text", deps=[FieldDep(feature_key=VideoFeature.spec.key, fields=["audio"])])])

class TranscriptFeature(BaseTranscriptFeature, spec=transcript_spec):
    path: str
```

Voilà!

The [Data Versioning](data-versioning.md) docs explain more about this system.

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
