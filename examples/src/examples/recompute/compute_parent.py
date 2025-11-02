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
    MetaxyConfig,
    get_feature_by_key,
    load_features,
)

load_features()

# feature showcase: get feature classes by key
parent_key = FeatureKey(["examples", "parent"])
ParentFeature = get_feature_by_key(parent_key)

# Get metadata store from metaxy.toml config
with MetaxyConfig.load().get_store() as store:
    # Save feature graph snapshot, normally this should be done in CI/CD before running the pipeline
    result = store.record_feature_graph_snapshot()

    snapshot_version = result.snapshot_version

    _ = result.already_recorded

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
            "provenance_by_field": [
                {"embeddings": "v1"},
                {"embeddings": "v2"},
                {"embeddings": "v3"},
            ],
        },
        schema={
            "sample_uid": pl.UInt32,
            "raw_data": pl.Utf8,
            "provenance_by_field": pl.Struct({"embeddings": pl.Utf8}),
        },
    )
    store.write_metadata(ParentFeature, parent_metadata)

    print(f"Written {len(parent_metadata)} rows for feature {parent_key}")
