---
title: 'Metaxy: GPU-Conscious Feature Metadata Management for Rapid ML Experimentation'
tags:
  - Python
  - metaxy
  - GPU
  - featurestore
  - featuregraph
  - metadata
authors:
  - name: Daniel Gafni
    # orcid: 0000-0000-0000-0000
    affiliation: "1"
  - name: Georg Heiler
    orcid: 0000-0002-8684-1163
    affiliation: "2, 3"
affiliations:
  - name: Complexity Science Hub Vienna (CSH)
    index: 2
  - name: Austrian Supply Chain Intelligence Institute (ASCII)
    index: 3
  - name:  Anam.ai
    index: 1
date: 2025-10-31
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# <https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Journal of Open Source Software
---

# Summary

Modern machine learning demands rapid experimentation across model architectures, hyperparameters, and feature transformations.
Research teams must iterate quickly to remain competitive, yet each experiment requires careful lineage tracking to ensure reproducibility.
This tension intensifies in GPU-accelerated pipelines where compute minutes cost 10–100× more than traditional CPU workloads, making redundant recomputation prohibitively expensive.
Teams face a dilemma: iterate quickly and lose reproducibility, or maintain rigorous tracking at the cost of velocity.

Metaxy resolves this conflict through topological dependency graphs with record-level versioning.
Unlike table-level orchestrators, Metaxy tracks versions for individual data instances and computes which specific records require reprocessing based on upstream changes.
The library resolves incremental updates deterministically in the metadata layer before any GPU execution, eliminating redundant computation while preserving complete lineage.

The system integrates with lakehouse storage, more than twenty SQL backends through Ibis (@ibis), and backend-agnostic data frames via Narwhals (@narwhals).
Orchestration platforms such as Dagster (@dagster) and Ray (@ray) consume Metaxy version diffs to gate GPU workloads behind concrete evidence of metadata changes.
An automated test suite ensures version stability and deterministic change propagation across heterogeneous execution environments.

# Statement of Need

Machine learning research requires rapid iteration on feature definitions, model architectures, and training strategies.
Teams must experiment freely to discover what works, yet reproducibility demands rigorous lineage tracking.
Traditional approaches force a trade-off: either rerun entire pipelines to guarantee correctness, sacrificing iteration speed, or skip proper tracking to move quickly, sacrificing reproducibility.

This dilemma is particularly acute for multimodal pipelines processing video, audio, and high-resolution imagery.
When a researcher updates an audio processing algorithm, should the system recompute face detections that depend only on video frames?
Existing orchestrators operate at table granularity, treating entire feature tables as atomic units.
This coarse granularity forces unnecessary recomputation because the system cannot distinguish which fields within a record have changed.
The resulting waste becomes prohibitive when GPU compute costs 10–100× more per hour than traditional CPU workloads.

Metaxy addresses this challenge through topological dependency graphs operating at record granularity.
Researchers define features as Pydantic models specifying which upstream fields each computation depends on.
The system constructs a dependency graph where nodes represent feature fields and edges represent data flow.
For each record, Metaxy computes version hashes based on upstream record versions and code versions, propagating changes through the graph topology.
When resolving incremental updates, the system returns only those specific records whose upstream dependencies have changed.

This approach delivers both rapid experimentation and reproducibility.
Researchers modify feature definitions freely, knowing the system will identify exactly which records require reprocessing.
The metadata layer preserves complete lineage, allowing retrospective audits of any experiment.
By computing diffs before execution, teams quantify the impact of code changes and defer updates that do not justify their computational cost.

The target audience includes machine learning engineers working with multimodal data, MLOps teams managing production feature pipelines, and research labs operating under compute budgets.
Metaxy supports prototyping on laptops with DuckDB and scales to production clusters with ClickHouse, BigQuery, or any backend supported by Ibis.

# System Overview

Metaxy represents features as declarative Pydantic (@pydantic) models organized into a directed acyclic graph.
Each feature declares its identifier columns, computed fields, and topological dependencies on upstream feature fields.
The system constructs a global dependency graph at initialization where nodes represent feature fields and edges represent data flow, enabling downstream version propagation before any data processing occurs.

Version computation operates at record granularity through hierarchical hashing.
Field code versions are user-specified strings marking algorithmic changes.
For each record, field versions combine code versions with upstream record field versions through deterministic hashing, propagating changes along graph edges.
Feature versions aggregate field versions defined on that feature.
Record versions map each data instance to the specific upstream field versions that produced it, creating a precise per-record provenance trail.

This record-level granularity distinguishes Metaxy from table-level orchestrators.
When an upstream field changes for a subset of records, only those downstream records that transitively depend on the changed field are marked for recomputation.
Records whose dependencies remain unchanged are skipped, eliminating redundant work.

Metadata persistence uses pluggable storage backends.
DuckDB provides embedded local storage for prototyping.
ClickHouse, BigQuery, and any Ibis-supported database scale to production workloads.
Narwhals supplies a backend-agnostic dataframe interface, allowing users to work with Pandas, Polars, or Ibis without vendor lock-in.
The system pushes version computation into the metadata store through SQL, avoiding expensive data transfers when resolving incremental updates.

Developers define features through Python classes, visualize dependency graphs through the CLI, and integrate with orchestrators via the Python API.
Orchestration platforms consume record-level metadata diffs to make scheduling decisions, gating GPU-intensive assets behind concrete evidence of upstream changes.
The append-only storage design preserves complete historical lineage, enabling retrospective experiment audits and migration detection.

# Minimal usage example

A typical workflow resolves metadata differences before launching GPU workloads.
After initializing feature definitions, practitioners ask the metadata store which samples require recomputation, run the expensive jobs for that subset, and append fresh metadata for the new versions.

```python
from metaxy import init_metaxy
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from voice.features import VoiceDetection

init_metaxy()

with DuckDBMetadataStore("metaxy.duckdb") as store:
    diff = store.resolve_update(VoiceDetection)
    if diff.added or diff.changed:
        results = run_voice_detection(diff)  # user-defined GPU workload
        store.write_metadata(VoiceDetection, results)
```

This flow keeps recomputation focused on the minimum viable subset while maintaining end-to-end lineage.
Because topological dependencies are encoded in record-level versions, downstream aggregations only rerun when the relevant upstream fields actually changed.

# Evaluation

Correctness is validated through automated tests covering three critical properties.
First, version stability tests (`tests/test_feature_version.py`, `tests/test_snapshot_version_stability.py`) ensure that hash computation is deterministic across Python interpreter versions and code refactorings.
Second, incremental update tests (`tests/test_metadata_store.py`) verify that `resolve_update` returns exactly those records whose upstream dependencies have changed through the topological graph, neither missing updates nor triggering false positives.
Third, cross-backend tests exercise the same feature definitions against DuckDB, ClickHouse, DeltaLake, and Lance to confirm consistent record-level metadata semantics.

Example pipelines demonstrate the system's impact on multimodal workflows.
A video processing pipeline (`examples/overview`) defines features for audio transcription and face detection with topological dependencies between fields.
When only the audio processing algorithm changes, the system correctly schedules transcription updates for affected records while leaving face detection metadata unchanged.
This record-level selective recomputation is the core value proposition: only records depending on the modified field incur GPU costs.

# Impact and Future Work

Metaxy enables fearless experimentation without sacrificing reproducibility.
Researchers modify feature definitions freely, knowing the system will identify exactly which records require reprocessing.
The append-only lineage design ensures that every experiment remains reproducible, addressing a critical gap in machine learning research.
By computing metadata diffs before execution, teams can iterate rapidly while maintaining complete audit trails.

This approach also addresses GPU economics.
Record-level selective execution eliminates redundant computation, making expensive GPU workloads financially viable.
Teams quantify the cost-benefit of each pipeline change and defer updates that do not justify their computational expense.

The system bridges prototyping and production through vendor-neutral abstractions.
Researchers define features once and deploy them across DuckDB on laptops, ClickHouse in data centers, or BigQuery in the cloud without modifying feature code.
This portability lowers the operational barrier to disciplined metadata management, making reproducibility the default rather than an afterthought.

Future work will extend orchestration integrations and cost modeling.
Planned features include native Dagster sensors that trigger downstream assets based on Metaxy metadata diffs, direct cost estimation by propagating per-sample compute costs through the dependency graph, and integration with GPU cluster telemetry to track actual resource consumption against predicted metadata changes.

The project welcomes contributions at https://github.com/anam-org/metaxy.
Documentation is available at https://docs.metaxy.io.

# Acknowledgements

We thank the maintainers of the Ibis, Narwhals, Pydantic, and Dagster communities for the foundational tooling Metaxy builds upon, and the contributors who tested early releases across heterogeneous hardware.
Funding and in-kind support were provided by [Anam](https://anam.ai/) the Complexity Science Hub Vienna and the Austrian Supply Chain Intelligence Institute.

# References
