"""Pipeline for recompute example.

This script represents the actual user pipeline that:
1. Loads upstream data
2. Computes parent feature (embeddings)
3. Computes child feature (predictions) using parent embeddings

The VERSION environment variable determines which feature versions are loaded.
Run with VERSION=1 initially, then VERSION=2 to see recomputation.
"""

import metaxy as mx
import polars as pl
from metaxy.metadata_store.system import SystemTableStorage

# Initialize metaxy (loads config and discovers features)
config = mx.init_metaxy()

# feature showcase: get feature definitions by key
parent_key = mx.FeatureKey(["examples", "parent"])

with config.get_store() as store:
    result = SystemTableStorage(store).push_graph_snapshot()

    snapshot_version = result.snapshot_version

    print(f"Graph snapshot_version: {snapshot_version}")

    # Check if metadata already exists for current feature_version (avoid duplicates)
    try:
        existing = store.read(parent_key, with_feature_history=False)
        feature_version = mx.current_graph().get_feature_version(parent_key)
        if existing.collect().shape[0] > 0:
            print(
                f"Metadata already exists for feature {parent_key} (feature_version: {feature_version[:16]}...)"
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
    store.write(parent_key, parent_metadata)

    print(f"Written {len(parent_metadata)} rows for feature {parent_key}")
