# Solving multimodal Pipelines with Dagster and Metaxy

My name is Daniel Gafni, and I'm an MLOps engineer at [Anam](https://anam.ai?utm_source=dagster&utm_medium=blog&utm_campaign=dagster-blog-post).

At [Anam](https://anam.ai?utm_source=dagster&utm_medium=blog&utm_campaign=dagster-blog-post), we are making a platform for building real-time interactive avatars. One of the key components powering our product is our own video generation model.

We train it on custom training datasets that require all sorts of pre-processing of video and audio data. We extract embeddings with ML models, use external APIs for annotation and data synthesis, and so on. Of course, we use [Dagster](https://dagster.io) to orchestrate these steps as [data assets](https://docs.dagster.io/guides/build/assets).

In this blog post, we're going to share how we used Dagster together with our new open source [Metaxy](https://metaxy.io) framework to solve **sample-level versioning** for these pipelines.

Metaxy connects orchestrators such as [Dagster](https://dagster.io) that usually operate at table (or asset) level with low-level processing engines (such as [Ray](https://ray.io)), allowing us to process exactly the samples that *have to* be processed at each step and *not a sample more*.

## The Little Change

A few months ago we decided to introduce a little change into the data preparation pipeline. At that time, we were using a custom data versioning system which tracked a version for each sample. The system could compute a fingerprint for each step based on a manually specified `code_version` on the step and the upstream steps (sounds familiar?). The only difference with [Dagster's data versioning](https://docs.dagster.io/guides/build/assets/asset-versioning-and-caching) was the granularity level: we computed it for each *row* in the dataset, while Dagster only supports asset-level versioning.

The change we wanted to introduce was very simple: we wanted to crop our videos at a new resolution. This implied changing the `code_version` of the cropping stage, and the downstream steps would be re-computed automatically. However, we also noticed an unpleasant outcome. Right after the cropping step, our pipeline branched into two regions:

![pipeline-branches](./assets/pipeline.svg)

Half of the downstream steps were not even using the cropped video frames. They only operated on the audio part. But our data versioning system was unaware of this detail and would re-compute them anyway. This means running our custom audio ML models on the entire training dataset: **very expensive and absolutely unnecessary**.

This was the moment I realized there was something wrong with our naive approach to data versioning. The idea of Metaxy - the project we are going to discuss in this blog post - was born.

## A Glimpse Into Multimodal Data

As the software world is being eaten by AI, more teams and organizations are starting to interact with multimodal data pipelines. Unlike traditional data engineering workflows, these pipelines are not dealing just with tables, but also with text, images, audio, videos, vector embeddings, medical data, and so on.

Multimodal data pipelines can be very unique, with requirements and complexity varying from use case to use case. Whether calling AI APIs over HTTP, running local ML inference, or simply invoking `ffmpeg`, there is something in common: compute and I/O gets expensive very quickly.

When traditional (tabular) data pipelines are re-executed, they typically don't cost much. Sure, Big Data exists, and Spark jobs can query petabytes of tabular data, but in reality very few teams actually run into these issues. That's a big reason behind the [Small Data movement's](https://motherduck.com/blog/small-data-manifesto/) success: the median Snowflake scan reads less than 100MB of data, and 80% of organizations have less than 10TB of data! Therefore, re-running a tabular pipeline is usually fine. It's also much easier to do than to implement incremental processing.

Multimodal pipelines are a whole different beast. They require a few orders of magnitude more compute and data movement. Accidentally re-executed your Whisper voice transcription step on the whole dataset? Congratulations: $10k just wasted!

That's why with multimodal pipelines, implementing incremental approaches is a requirement rather than an option. And it turns out, it's damn complicated.

## Introducing Metaxy

Metaxy is the missing piece connecting the Dagster world - that's only aware of datasets (or their partitions) - with the compute world, which has to deal with individual samples.

Metaxy has two features that make it unique:

1. It is able to track partial data updates.

2. It is agnostic to infrastructure and can be plugged into any data pipeline written in Python.

It implements the same approach Dagster takes for computing data versions, but extends it:

- to work in batches, for millions of rows at a time

- to run in a remote database or locally

- to be agnostic to dataframe engines or DBs

- to be aware of *data fields*: instead of being a string, each data version in Metaxy is actually a dictionary.

## Data Fields

One of the main goals of Metaxy is to enable granular partial data versioning, so that partial updates are recognized correctly. In Metaxy, alongside the normal *metadata columns*, users can also define and version *data fields*:

![feature](./assets/feature.svg)

Data fields can be arbitrary and designed as users see fit. The key takeaway is that data fields describe *data* (e.g. `mp4` files) and not tabular *metadata*.

Then, you can run a few lines of Python code:

<!-- skip next -->
```python
with store:
    increment = store.resolve_update("video/face_crop")
```

which does a lot of work behind the scenes:

- joining state tables for upstream steps

- computing expected data versions for each row - **this is the most complicated step**. It is complicated because each version is a dictionary, and each field in the dictionary may depend on its own subset of upstream fields

- loading the state table for the step being resolved and comparing versions with the expected ones

- returning *new*, *stale* and *orphaned* samples to the user

Once the user gets the `increment` object (by the way - it can be lazy!), they can decide what to do with each category of samples: typically *new* and *stale* are processed again, while *orphaned* may be deleted.

Metaxy solves the "little change" problem by being aware of *partial data dependencies*.

Consider 3 Metaxy *features* (that's how Metaxy calls the data produced at each step): video files (`video/full`), Whisper transcripts (`transcript/whisper`), and video files cropped around the face (`video/face_crop`):

![Metaxy Anatomy](./assets/anatomy.svg)

Separate information paths for audio and frames are color-coded. Notice how there are clear *field-level*, or *partial* data dependencies between features. Each *field version* is computed from the versions of the fields it depends on. Field versions from the same feature are then combined together to produce a *feature version*.

It is obvious that the `text` *field* of the `transcript/whisper` feature only depends on the `audio` *field* of the `video/full` feature. If we decided to resize `video/full`, then `transcript/whisper` doesn't have to be recomputed.

Metaxy detects these kinds of "irrelevant" updates and skips recomputation for downstream features that do not have *fields* affected by upstream changes. This is achieved by recording a separate data version for each field of every sample of every feature:

| id        | metaxy_provenance_by_field                    |
| --------- | --------------------------------------------- |
| video_001 | `{"audio": "a7f3c2d8", "frames": "b9e1f4a2"}` |
| video_002 | `{"audio": "d4b8e9c1", "frames": "f2a6d7b3"}` |
| video_003 | `{"audio": "c9f2a8e4", "frames": "e7d3b1c5"}` |
| video_004 | `{"audio": "b1e4f9a7", "frames": "a8c2e6d9"}` |

### The challenge of composability

As was mentioned earlier, incremental pipelines are diverse: they often run in unique environments, require specific infrastructure, different cloud providers (including Neoclouds), or scaling engines such as [Ray](https://www.ray.io/) or [Modal](https://modal.com/).

One of the goals of Metaxy was to be as versatile and agnostic as Dagster and support this variety of use cases. Metaxy had to be *pluggable* in order to be usable by different users and organizations.

And it turns out, this is possible! 95% of metadata management work done by Metaxy is implemented in a way that's agnostic to databases, and can even run locally with [Polars](https://pola.rs/) or [DuckDB](https://duckdb.org/)!

This is only possible due to **the incredible amount of work** that has been put into the [Ibis](https://ibis-project.org/) and [Narwhals](https://narwhals-dev.github.io/narwhals/) projects. Ibis implements the same Python interface (*not* a typical DataFrame API) for 20+ databases, while Narwhals does the same for different DataFrame engines (Pandas, Polars, DuckDB, Ibis, and more), converging everything to a subset of the Polars API.

Narwhals (or Polars) expressions are the GOAT for programmatic query building. Most of Metaxy's *versioning engine* is implemented in Narwhals expressions, while a few narrow parts had to be pushed back to specific backends.

The importance of this cannot be stressed enough. Entire new generations of *composable data tooling* can be built on top of Narwhals - and of course, none of this would be possible without [Apache Arrow](https://arrow.apache.org/).

## Metaxy and Dagster

Of course, Metaxy was designed to be used with Dagster. Not only did it take inspiration from Dagster's data versioning design, but it also naturally inherited some of the other properties of Dagster: being declarative and asset-oriented. Just take a look at this API, which should look extremely familiar to any Dagster user:

```python
import metaxy as mx

spec = mx.FeatureSpec(
    key="video/crop",
    id_columns=["id"],
    deps=["video/raw"],
    description="Videos cropped to 720x480.",
    metadata={"team": "ML"},
)
```

Like Dagster, Metaxy has a declarative DSL for defining DAGs, where nodes represent data. They are called *Features* in Metaxy. Metaxy *Features* directly map into Dagster *Assets*.

The `spec` from above can be attached to a *feature class*:

<!-- skip next -->
```python
class VideoCrop(mx.BaseFeature, spec=spec):
    path: str
    duration: float
    frame_count: int
```

Now, these *feature definitions* can be trivially integrated with Dagster assets:

<!-- skip next -->
```python
import dagster as dg
import metaxy as mx

from metaxy.ext.dagster import metaxify


@metaxify()
@dg.asset(metadata={"metaxy/feature": "video/crop"})
def video_crops(store: dg.ResourceParam[mx.MetadataStore]):
    with store:
        changes = store.resolve_update("video/crop")

    with mx.BufferedMetadataWriter(store) as writer:
        for sample in changes.new:
            # handle each sample here, e.g. run `ffmpeg` to crop videos
            # or do something else

            # once done, submit metadata to Metaxy
            # BufferedMetadataWriter flushes them automatically
            writer.put({"video/crop": results})
```

That's it!

The `@metaxify` decorator here does a lot of heavy lifting, injecting all the information available to Metaxy - and Metaxy knows everything about the feature/asset graph - into the otherwise bare-bones Dagster asset. It gets correct lineage, Dagster's `code_version`, table schema metadata, and more.

Metaxy also integrates with Dagster in a few other ways. A notable mention would be the `MetaxyIOManager`, which allows doing I/O with any of the Metaxy-supported metadata stores, such as DuckDB, DeltaLake, ClickHouse - and you can even use the same code while swapping them for development and production on demand!

Because this integration is so non-invasive and simple, Metaxy acts as an extension to Dagster for orchestrating individual samples in multimodal datasets.

## Conclusion

Until now, managing individual files in Dagster pipelines has been a very cumbersome task. Metaxy makes this easy, allowing Dagster users to focus on transformations.
Dagster and Metaxy elegantly complement each other: the former manages asset-level orchestration and kicks off pipelines calling Metaxy code, while the latter handles row-level orchestration with sub-sample granularity.

Metaxy can also be used outside of Dagster! And surely we're going to discover a lot more of what's possible with it.

Read our docs [here](https://docs.metaxy.io/) and `uv pip install metaxy[dagster]`!

We are thrilled to help more users solve their metadata management problems with Metaxy. Please do not hesitate to reach out on [GitHub](https://github.com/metaxy-io/metaxy)!

## Acknowledgments

Thanks to [Georg Heiler](https://github.com/geoHeil) for contributing to the project with discussions and code, and thanks to the open source projects Metaxy is built on: [Narwhals](https://narwhals-dev.github.io/narwhals/), [Ibis](https://ibis-project.org/), and others.
