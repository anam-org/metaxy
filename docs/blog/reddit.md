My name is Daniel Gafni, and I'm an MLOps engineer at [Anam](https://anam.ai).

## What My Project Does

[Metaxy](https://metaxy.io) is a pluggable metadata layer for building multimodal Data and ML pipelines. Metaxy manages and tracks metadata across complex computational graphs and implements sample and sub-sample versioning.

[Metaxy](https://metaxy.io) sits in between high level orchestrators (such as [Dagster](https://dagster.io)) that usually operate at table level and low-level processing engines (such as [Ray](https://ray.io)), passing the exact set of samples that have to be (re) computed to the processing layer and *not a sample more*.

## Target Audience

ML and data engineers who build multimodal custom data and ML pipelines and need incremental capabilities.

## Comparison

No exact alternatives exist. [Datachain](https://datachain.ai/) is a honorable mention, but it's a feature rich end-to-end platform, while Metaxy aims to be more minimalistic and pluggable (and only handles metadata, not compute compute).

## Background

At [Anam](https://anam.ai), we are making a platform for building real-time interactive avatars. One of the key components powering our product is our own video generation model.

We train it on custom training datasets that require all sorts of pre-processing of video and audio data. We extract embeddings with ML models, use external APIs for annotation and data synthesis, and so on. 

We encountered significant challenges with implementing efficient and versatile sample-level versioning (or caching) for these pipelines, which led us to develop and open-source [Metaxy](https://metaxy.io): the framework that solves metadata management and **sample-level versioning** for multimodal data pipelines.

When a traditional (tabular) data pipeline gets re-executed, it typically doesn't cost much. Multimodal pipelines are a whole different beast. They require a few orders of magnitude more compute, data movement and AI tokens spent. Accidentally re-executed your Whisper voice transcription step on the whole dataset? Congratulations: $10k just wasted!

That's why with multimodal pipelines, implementing incremental approaches is a requirement rather than an option. And it turns out, it's damn complicated.

## Introducing Metaxy

Metaxy is the missing piece connecting traditional orchestrators (such as [Dagster](https://dagster.io/) or [Airflow](https://airflow.apache.org/)) that usually operate at a high level (e.g., updating tables) with the sample-level world of multimodal pipelines.

Metaxy has two features that make it unique:

1. It is able to track partial data updates.

2. It is agnostic to infrastructure and can be plugged into any data pipeline written in Python.

Metaxy's versioning engine:

- operates in batches, easily scaling to millions of rows at a time.

- runs in a powerful remote database or locally with Polars or DuckDB.

- is agnostic to dataframe engines or DBs.

- is aware of *data fields*: Metaxy tracks a dictionary of versions for each sample.

We have been dogfooding Metaxy at Anam since December 2025. We are running millions of samples through Metaxy. All the current Metaxy functionality has been built for our data pipeline and is used there.

### AI Disclaimer

Metaxy has been developed with the help of AI tooling (mostly Claude Code). However, it should not be considered a vibe-coded project: the core design ideas are human, AI code has been ruthlessly reviewed, we run a very comprehensive test suite with 85% coverage, all the docs have been hand-written (seriously, I hate AI docs), and /u/danielgafni has been working with multimodal pipelines for three years before making Metaxy. A great deal of effort and passion went into Metaxy, especially into user-facing parts and the docs.

## More on Metaxy

Read our [blog post](https://anam.ai/blog/metaxy), [Dagster + Metaxy blog post](https://dagster.io/blog/building-real-time-interactive-avatars-with-metaxy), Metaxy [docs](https://docs.metaxy.io/), and `uv pip install metaxy`!

We are thrilled to help more users solve their metadata management problems with Metaxy. Please do not hesitate to reach out on [GitHub](https://github.com/metaxy-io/metaxy)!
