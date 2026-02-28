---
title: 'Metaxy: Record-Level Feature Metadata Management for GPU-Accelerated ML Pipelines'
tags:
  - Python
  - machine learning
  - metadata
  - feature engineering
  - reproducibility
  - GPU
authors:
  - name: Daniel Gafni
    affiliation: "1"
  - name: Georg Heiler
    orcid: 0000-0002-8684-1163
    affiliation: "2, 3"
affiliations:
  - name: Anam
    index: 1
  - name: Complexity Science Hub Vienna (CSH)
    index: 2
  - name: Austrian Supply Chain Intelligence Institute (ASCII)
    index: 3
date: 2026-02-16
bibliography: paper.bib
---

# Summary

Software that processes large datasets often repeats expensive computations when any part of the input data or processing logic changes.
In machine learning pipelines that handle video, audio, and images, these computations run on Graphics Processing Units (GPUs) that cost 10 to 100 times more per hour than standard processors.
A small change to one processing step can trigger unnecessary recomputation of unrelated steps, wasting both time and money.

Metaxy is a Python library that tracks which specific data records need reprocessing after a change, rather than rerunning entire datasets.
It builds a dependency graph that connects individual data fields across processing steps.
When a researcher modifies one step, Metaxy identifies exactly which records are affected and which can be skipped.
This selective approach eliminates redundant GPU work while preserving complete lineage for reproducibility.

The library integrates with various backends through Ibis (@ibis) and supports multiple dataframe engines via Narwhals (@narwhals).
Orchestration platforms such as Dagster (@dagster) and Ray (@ray) consume Metaxy's record-level diffs to schedule only the necessary GPU workloads.

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

![Metaxy tracks separate version hashes for each field of every record. In this example, the `audio` and `frames` fields of a video feature propagate independently through downstream features. A change to the audio processing algorithm only triggers recomputation of audio-dependent downstream fields, leaving frame-dependent fields unchanged.\label{fig:anatomy}](paper-assets/anatomy.svg)

# State of the Field

Several tools address aspects of feature management, but none provide field-level dependency tracking at record granularity.
DVC (@dvc) versions datasets at the file level, treating each file as an opaque artifact without tracking individual records or fields within it.
Feast (@feast) serves precomputed feature values at inference time but does not model dependency graphs between features or detect which records need recomputation after upstream changes.
Hamilton (@hamilton) traces column-level lineage through Python function composition, yet operates at table granularity and cannot identify which specific records within a column are affected by a change.
DataChain (@datachain) bundles managed compute with metadata management, coupling storage and execution into a single system.

Metaxy fills the gap by separating metadata from compute: it tracks field-level dependencies at record granularity and propagates version changes topologically through the dependency graph.
This separation allows teams to integrate Metaxy with any compute framework, such as Ray (@ray) or Dagster (@dagster), while retaining precise control over which records require reprocessing.

# Software Design

Metaxy represents features as declarative Pydantic models organized into a directed acyclic graph.
Each feature declares its identifier columns, computed fields, and dependencies on upstream feature fields.
The system constructs a global dependency graph at initialization, enabling downstream version propagation before any data processing occurs.

Version computation operates at record granularity through hierarchical hashing.
Field code versions are user-specified strings that mark algorithmic changes.
For each record, field versions combine code versions with upstream record field versions through deterministic hashing.
Feature versions aggregate field versions, and record versions map each data instance to the specific upstream versions that produced it, creating a per-record provenance trail.

This record-level granularity is what distinguishes Metaxy from table-level orchestrators.
When an upstream field changes for a subset of records, only those downstream records that transitively depend on the changed field are marked for recomputation.
Records with unchanged dependencies are skipped.

Metadata persistence uses pluggable storage backends.
DuckDB provides embedded storage for prototyping; ClickHouse, BigQuery, Delta Lake, Ducklake, LanceDB and Postgresql scale to production workloads.
Narwhals supplies a backend-agnostic dataframe interface, allowing users to work with Pandas, Polars, or Ibis interchangeably.
The system pushes version computation into the metadata store through SQL, avoiding expensive data transfers.

# Research Impact

Correctness is validated through automated tests covering three properties.
Version stability tests ensure that hash computation is deterministic across Python interpreter versions and code refactorings.
Incremental update tests verify that the system returns exactly those records whose upstream dependencies changed, neither missing updates nor triggering false positives.
Cross-backend tests exercise the same feature definitions against all supported backends to confirm consistent metadata semantics across storage engines.

Example pipelines demonstrate the system's impact on multimodal workflows.
A video processing pipeline defines features for audio transcription and face detection with field-level dependencies.
When only the audio processing algorithm changes, the system correctly schedules transcription updates for affected records while leaving face detection metadata unchanged.
This selective recomputation is the core value proposition: only records depending on the modified field incur GPU costs.

Metaxy has been running in production at Anam for processing millions of training samples across multimodal video pipelines.
Record-level selective execution eliminates redundant computation, making expensive GPU workloads financially viable for iterative research.
The append-only lineage design ensures that every experiment remains reproducible, addressing a critical gap in machine learning workflows.

The system bridges prototyping and production through vendor-neutral abstractions.
Researchers define features once and deploy them across DuckDB on laptops, ClickHouse in data centers, or BigQuery in the cloud without modifying feature code.
This portability lowers the barrier to disciplined metadata management, making reproducibility the default rather than an afterthought.

The project welcomes contributions at [https://github.com/anam-org/metaxy](https://github.com/anam-org/metaxy).
Documentation is available at [https://docs.metaxy.io](https://docs.metaxy.io).

# AI Usage Disclosure

Generative AI tools were used during development for code completion and documentation drafting. All AI-generated content was reviewed and refined by the authors. Most of the documentation was even written by hand.

# Acknowledgements

We thank the maintainers of the Ibis, Narwhals, Pydantic, and Dagster communities for the foundational tooling Metaxy builds upon, and the contributors who tested early releases across heterogeneous hardware.
Funding and in-kind support were provided by Anam, the Complexity Science Hub Vienna, and the Austrian Supply Chain Intelligence Institute.

# References
