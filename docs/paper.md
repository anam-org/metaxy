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
    orcid: 0000-0000-0000-0000
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

Metaxy delivers reproducible feature metadata management for GPU-intensive machine-learning pipelines.
By encoding features as static graphs with field-level dependencies, the library resolves incremental updates deterministically before any expensive recomputation.
This advance keeps GPU experimentation lean by scheduling only the samples whose metadata actually changed, trimming both energy usage and iteration time.

Metaxy packages its approach as a permissive Python library with a CLI and documentation for applied researchers and practitioners.
The project integrates with lakehouse storage, more than twenty SQL backends through Ibis [@ibis], and backend-agnostic data frames via Narwhals [@narwhals] to keep deployments vendor-neutral.
The possibility of coupling with orchestrators such as Dagster [@dagster] and Ray [@ray] allows teams to feed version diffs directly into smart compute management policies, deferring GPU work that is not justified by metadata change.
An automated test suite protects version stability and ensures precise change propagation, making the tool reliable in production-like workflows.

# Statement of Need

GPU feature stores face distinct cost and agility pressures compared with traditional ETL pipelines.
The high marginal price of GPU minutes and the volatility of experimental feature definitions punish redundant recomputation, yet existing feature stores often target batch ETL on CPUs with coarse-grained dependency tracking.
Research teams therefore resort to ad-hoc metadata tables or manual notebooks, leading to wasteful duplication and irreproducible experiment state.

Metaxy targets this gap by coupling fine-grained feature lineage with resource-aware execution planning.
Its field-level dependency system and sample-aware versioning allow teams to preview compute deltas, quantify cost, and defer unnecessary GPU jobs without sacrificing traceability.
By unifying metadata capture across prototyping laptops and production clusters, the project lowers the barrier to disciplined experimentation for resource-constrained labs.

# System Overview

Metaxy encodes features as declarative Pydantic models [@pydantic] bound to a global feature graph.
Each feature declares its identifiers, fields, and dependencies, enabling the graph to determine downstream impacts of every change before execution.
Deterministic hashes summarize field, feature, and snapshot state, creating a reproducible contract between experimentation and deployment.

Metadata persistence is provided by pluggable stores, including DuckDB, ClickHouse, and any backend Ibis supports, while Narwhals supplies a backend-agnostic dataframe interface.
By computing dependency-aware versions entirely inside the configured store, Metaxy keeps metadata close to the data and avoids shuttling large intermediate tables.
The append-only design preserves historical lineage so users can audit experiments long after data has shifted.

Developers interact through a Python API, a CLI for graph visualization and inspection, and in the future integrations with orchestrators such as Dagster [@dagster] and Ray [@ray].
In these environments, Metaxy diffs can be ingested as scheduling inputs so that compute graphs prioritize samples with the highest impact, gracefully throttling or cancelling redundant GPU jobs.
Syntactic sugar for feature definitions, comprehensive type hints, and testing helpers keep authoring friction low, allowing teams to onboard quickly without sacrificing rigor.

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
Because field-level dependencies travel with the diff object, downstream aggregations only rerun when the relevant upstream fields actually changed.

# Evaluation

We exercise versioning invariants through a comprehensive automated test suite.
Unit and regression tests such as `tests/test_feature_version.py` and `tests/test_snapshot_version_stability.py` confirm that hashes remain stable across code refactors, while `tests/test_metadata_store.py` verifies consistent behavior for append-only writes, cross-environment fallbacks, and dependency resolution.
These checks ensure that the metadata plan matches the independent recompute results users would obtain in production.

Metaxy also ships executable examples that simulate multimodal media pipelines, demonstrating how field-level dependencies curtail unnecessary recomputation.
These scenarios show orchestration layers gating GPU execution on concrete, versioned diffs rather than heuristics.

# Impact and Future Work

Metaxy reduces the operational waste of GPU experimentation by grounding complex pipelines in reproducible metadata lineage.
Labs can iterate on feature definitions freely, knowing that only the affected samples will be recomputed and that historical runs remain explorable.
The project thus builds a bridge between exploratory research culture and the accountability expected in production data science, while slotting naturally into orchestrators that already govern job priorities.

Future work will improve ergonomics and deepen resource awareness.
Planned extensions include richer scheduling hints for workload managers and tighter integrations with GPU cluster telemetry.

We are actively looking for issues, PRs and other contributions https://github.com/anam-org/metaxy.
Check out the documentation at https://docs.metaxy.io for more details.

# Acknowledgements

We thank the maintainers of the Ibis, Narwhals, Pydantic, and Dagster communities for the foundational tooling Metaxy builds upon, and the contributors who tested early releases across heterogeneous hardware.

# References
