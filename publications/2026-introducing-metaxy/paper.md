---
title: 'Metaxy: Record-Level Feature Metadata Management for Multimodal ML Pipelines'
tags:
  - Python
  - machine learning
  - metadata
  - feature engineering
  - reproducibility
  - data lineage
  - incremental computation
  - caching
  - multimodal
authors:
  - name: Daniel Gafni
    orcid: 0000-0003-1237-6876
    affiliation: "1"
  - name: Georg Heiler
    orcid: 0000-0002-8684-1163
    affiliation: "2, 3"
affiliations:
  - name: Anam, United Kingdom
    index: 1
  - name: Complexity Science Hub Vienna (CSH), Austria
    index: 2
  - name: Austrian Supply Chain Intelligence Institute (ASCII), Austria
    index: 3
date: 2026-02-16
bibliography: paper.bib
---

# Summary

Software that processes large datasets often repeats expensive computations when any part of the input data or processing logic changes.
Metaxy is about perfecting the art of doing nothing: only compute what changed, save time and money, and accelerate exploration.
In machine learning pipelines that handle video, audio, and images, these computations run on Graphics Processing Units (GPUs) that cost 10 to 100 times more per hour than standard processors[^gpu-cost].
A small change to one processing step can trigger unnecessary recomputation of unrelated steps, wasting both time and money.

[^gpu-cost]: AWS EC2 on-demand pricing, 2026-04; the exact ratio depends on the instance families compared, with common CPU-to-GPU comparisons falling in this range and top-end accelerators exceeding it.

Metaxy is a Python library that tracks which specific data records need reprocessing after a change, rather than rerunning entire datasets.
It builds a dependency graph that connects individual data fields across processing steps.
When a researcher modifies one step, Metaxy identifies exactly which records are affected and which can be skipped.
This selective approach lets downstream systems avoid redundant GPU work when a change does not affect a record, while preserving complete lineage for reproducibility.

The library integrates with various backends through Ibis (@ibis) and supports multiple dataframe engines via Narwhals (@narwhals).
Orchestration platforms such as Dagster (@dagster) and Ray consume Metaxy's record-level diffs to schedule only the necessary GPU workloads.

# Statement of Need

Machine learning practitioners iterate rapidly on feature definitions, model architectures, and training strategies.
Reproducibility demands that every experiment maintain a clear record of which data and code produced each result.
Traditional pipeline tools force a trade-off: rerun entire pipelines to guarantee correctness, or skip tracking to move quickly.

This trade-off is especially costly for multimodal pipelines.
Consider a video processing system that extracts both audio transcripts and face detections.
When a researcher improves the audio denoising algorithm, should the system recompute face detections that depend only on video frames?
Existing orchestrators operate at table granularity, treating feature tables as atomic units.
They cannot distinguish which fields within a record changed, forcing unnecessary recomputation of all downstream steps.

Metaxy resolves this problem through field-level dependency tracking at record granularity.
Researchers declare features as Python classes (@pydantic) that specify which upstream fields each computation depends on.
The system constructs a directed acyclic graph where nodes represent feature fields and edges represent data flow (\autoref{fig:anatomy}).
For each data record, Metaxy computes version hashes that combine code versions with upstream record versions, propagating changes along graph edges.
When resolving incremental updates, the system returns only those records whose upstream dependencies have actually changed.

This approach delivers both rapid experimentation and reproducibility.
Researchers modify feature definitions freely, and the system identifies exactly which records require reprocessing.
The metadata layer preserves complete lineage, enabling retrospective audits of any experiment.
By computing diffs before execution, teams quantify the impact of each change and defer updates that do not justify their computational cost.

The target audience includes ML engineers working with multimodal data, MLOps teams managing production feature pipelines, and research labs operating under compute budgets.

![Metaxy tracks separate version hashes for each field of every record. In this example, the `audio` and `frames` fields of a video feature propagate independently through downstream features. A change to the audio processing algorithm only triggers recomputation of audio-dependent downstream fields, leaving frame-dependent fields unchanged.\label{fig:anatomy}](assets/anatomy.svg)

# State of the Field

Several tools address aspects of feature management, but none, to our knowledge, provide field-level dependency tracking at record granularity as a standalone metadata layer.
DVC (@dvc) versions datasets at the file level, treating each file as an opaque artifact without tracking individual records or fields within it.
Feast (@feast) focuses on feature definition, materialization, and online serving. Recent Feast releases include DAG-based feature computation, but Feast does not expose field-level provenance for propagating version changes and deciding which downstream records require recomputation after an upstream modification.
Apache Hamilton (@hamilton) builds dataflows from Python functions and can report lineage over the resulting DAG, typically at the level of nodes, columns, and dataframe outputs. It does not maintain per-record, per-field provenance for selective downstream invalidation.
DataChain (@datachain) combines metadata management with a Python-native data processing framework and supports delta processing over new or changed records. However, its incremental model is tied to dataset processing within that framework rather than to a standalone field-level dependency graph that can drive recomputation across multiple compute backends.

Metaxy fills the gap by separating metadata from compute: it tracks field-level dependencies at record granularity and propagates version changes topologically through the dependency graph.
This separation allows teams to integrate Metaxy with any compute framework, such as Ray or Dagster, while retaining precise control over which records require reprocessing.

# Software Design

Metaxy represents features as declarative Pydantic models organized into a directed acyclic graph.
Each feature declares its identifier columns, computed fields, and dependencies on upstream feature fields.
The system constructs a global dependency graph at initialization, enabling downstream version propagation before any data processing occurs.

Version computation operates at two layers through deterministic hashing. Graph-level feature hashes combine field code versions with upstream structure once per code change, using a fixed `hashlib.sha256` implementation. Record-level hashes run once per data row, combining per-record identifiers with upstream record versions inside the metadata store via SQL; this function is configurable and defaults to `xxhash32` where the backend supports it, keeping hashing colocated with storage and avoiding expensive data transfers.
Field code versions are user-specified strings that developers bump to mark algorithmic changes. This is a deliberate design choice rather than an automatic-detection problem: a static analyzer cannot reliably distinguish a behavior-preserving refactor from a semantic change, and silent false-positive invalidation would trigger prohibitively expensive GPU recomputation. Metaxy therefore delegates this judgement to the author.
Feature versions aggregate field versions, and record versions map each data instance to the specific upstream versions that produced it, creating a per-record provenance trail.

This record-level granularity is what distinguishes Metaxy from table-level orchestrators.
When an upstream field changes for a subset of records, only those downstream records that transitively depend on the changed field are marked for recomputation.
Records with unchanged dependencies are skipped.

Metadata persistence uses pluggable, append-only storage backends: version entries are never overwritten, preserving the lineage needed for retrospective audits.
DuckDB provides embedded storage for prototyping; ClickHouse, BigQuery, Delta Lake, DuckLake, LanceDB and PostgreSQL scale to production workloads.
Narwhals supplies a backend-agnostic dataframe interface, allowing users to work with Pandas, Polars, or Ibis interchangeably.

# Performance

Per-record hashing and the downstream diff are executed as SQL inside the metadata store, so their throughput is bounded by the vectorised hash kernels of the backend rather than by Python.
The benchmark script `publications/2026-introducing-metaxy/benchmark.py` materialises a two-feature graph (a root with fields `audio` and `frames`, and a downstream leaf) against a fresh DuckDB store, measures `resolve_update` for the initial materialisation (`resolve_new`), then bumps the upstream `audio` provenance for 10% of records and measures the incremental diff (`resolve_stale`).
Medians over three runs on an Apple M2 Max (64 GiB RAM, Python 3.10.19, DuckDB 1.4.3, `xxhash64`) are reported in \autoref{tab:benchmark}; throughput scales linearly with `N` and saturates near $10^7$ records per second once the hash kernel dominates fixed setup costs.

: Median `resolve_update` wall-clock time on DuckDB for `N` records across a 2-feature graph (3 runs, 10% change fraction). \label{tab:benchmark}

|          N |  resolve_new (s) | resolve_stale (s) |   rows/s (new) |
|-----------:|-----------------:|------------------:|---------------:|
|     10,000 |            0.089 |             0.128 |        112,938 |
|    100,000 |            0.101 |             0.188 |        985,968 |
|  1,000,000 |            0.169 |             0.439 |      5,931,115 |
|  5,000,000 |            0.463 |             1.824 |     10,806,628 |
| 10,000,000 |            0.948 |             3.434 |     10,553,281 |

# Research Impact

Correctness is validated through automated tests covering three properties.
Version stability tests ensure that hash computation is deterministic across Python interpreter versions and code refactorings.
Incremental update tests verify that the system returns exactly those records whose upstream dependencies changed, neither missing updates nor triggering false positives.
Cross-backend tests exercise the same feature definitions against all supported backends to confirm consistent metadata semantics across storage engines.

Example pipelines demonstrate the system's impact on multimodal workflows.
A video processing pipeline defines features for audio transcription and face detection with field-level dependencies.
When only the audio processing algorithm changes, the system correctly schedules transcription updates for affected records while leaving face detection metadata unchanged.
This selective recomputation is the core value proposition: the metadata layer exposes the exact update set, so teams can schedule only affected records instead of maintaining that decision logic manually.

Metaxy has been running in production at Anam on multimodal video pipelines whose training corpora reach the low millions of samples.
Before Metaxy, achieving selective recomputation at this scale required manual metadata edits, ad-hoc overrides, and custom per-pipeline bookkeeping; Metaxy standardizes those decisions into a declarative model so that correct, auditable incremental updates become the default path rather than a bespoke engineering effort for each new feature.
Combined with the append-only metadata layer introduced above, every experiment remains reproducible, closing a critical gap in machine learning workflows.

The system bridges prototyping and production through vendor-neutral abstractions.
Researchers define features once and deploy them across DuckDB on laptops, ClickHouse in data centers, or BigQuery in the cloud without modifying feature code.
This portability lowers the barrier to disciplined metadata management, making reproducibility the default rather than an afterthought.

The project welcomes contributions at [https://github.com/anam-org/metaxy](https://github.com/anam-org/metaxy).
Documentation is available at [https://docs.metaxy.io](https://docs.metaxy.io).

# AI Usage Disclosure

Generative AI tools were used during development for code completion and documentation drafting.
All AI-generated content was reviewed and refined by the authors.
Large parts of the documentation were written by hand.

# Acknowledgements

We thank the maintainers of the Ibis, Narwhals, Pydantic, and Dagster communities for the foundational tooling Metaxy builds upon, and the contributors who tested early releases across heterogeneous hardware.
Funding and in-kind support were provided by Anam, the Complexity Science Hub Vienna, and the Austrian Supply Chain Intelligence Institute.

# References
