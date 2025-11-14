"""Compute parent feature (embeddings) - unchanged between versions."""

from pathlib import Path

import polars as pl

from metaxy import FeatureKey, get_feature_by_key, init_metaxy, load_features

# Load features
load_features()

# Get feature class
ParentFeature = get_feature_by_key(FeatureKey(["examples", "parent"]))

# Load upstream data
data_dir = Path("/tmp/migration_example_data")
upstream_data = pl.read_parquet(data_dir / "upstream_data.parquet")

# Get metadata store
config = init_metaxy()
with config.get_store() as store:
    print(f"📊 Computing {ParentFeature.spec().key.to_string()}...")
    print(f"  feature_version: {ParentFeature.feature_version()[:16]}...")

    # Simulate computing embeddings - same computation in both versions
    parent_data = pl.DataFrame(
        {
            "sample_uid": upstream_data["sample_uid"],
            "metaxy_provenance_by_field": [
                {"embeddings": f"embed_{sid}"} for sid in upstream_data["sample_uid"]
            ],
        }
    )

    store.write_metadata(ParentFeature, parent_data)
    print(f"✓ Materialized {len(parent_data)} samples")
