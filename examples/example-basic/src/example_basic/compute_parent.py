"""Pipeline for recompute example.

This script represents the actual user pipeline that:
1. Loads upstream data
2. Computes parent feature (embeddings)
3. Computes child feature (predictions) using parent embeddings

The VERSION environment variable determines which feature versions are loaded.
Run with VERSION=1 initially, then VERSION=2 to see recomputation.
"""

import polars as pl

from metaxy import (
    FeatureKey,
    get_feature_by_key,
    init_metaxy,
)
from metaxy.metadata_store.system import SystemTableStorage

# Initialize metaxy (loads config and discovers features)
config = init_metaxy()

# feature showcase: get feature classes by key
parent_key = FeatureKey(["examples", "parent"])
ParentFeature = get_feature_by_key(parent_key)

with config.get_store() as store:
    # Save feature graph snapshot, normally this should be done in CI/CD before running the pipeline
    result = SystemTableStorage(store).push_graph_snapshot()

    snapshot_version = result.snapshot_version

    print(f"Graph snapshot_version: {snapshot_version}")

    # Check if metadata already exists for current feature_version (avoid duplicates)
    try:
        existing = store.read_metadata(ParentFeature, current_only=True)
        if existing.collect().shape[0] > 0:
            print(
                f"Metadata already exists for feature {parent_key} (feature_version: {ParentFeature.feature_version()[:16]}...)"
            )
            print("Skipping write to avoid duplicates")
            exit(0)
    except Exception:
        # No existing metadata or feature not found, proceed with write
        pass

    parent_metadata = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "raw_data": ["sample_1_data", "sample_2_data", "sample_3_data"],
            "metaxy_provenance_by_field": [
                {"embeddings": "v1"},
                {"embeddings": "v2"},
                {"embeddings": "v3"},
            ],
        },
        schema={
            "sample_uid": pl.UInt32,
            "raw_data": pl.Utf8,
            "metaxy_provenance_by_field": pl.Struct({"embeddings": pl.Utf8}),
        },
    )
    store.write_metadata(ParentFeature, parent_metadata)

    print(f"Written {len(parent_metadata)} rows for feature {parent_key}")
